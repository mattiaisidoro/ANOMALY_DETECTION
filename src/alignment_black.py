"""
alignment_tappo4.py
=====================
Modulo per l'allineamento del "tappo tipo 4" (tappo opaco, senza marker
circolari da agganciare con Hough come nei tappi trasparenti).

APPROCCIO
---------
A differenza di align_cap_image / align_cap_image_small (che si basano su
una coppia di marker fisici rilevati via Hough per calcolare l'angolo, e
poi testano solo 2 ipotesi — dritto o capovolto — contro il template),
questo tappo non ha marker paired utilizzabili. L'orientamento viene
determinato tramite RICERCA ANGOLARE via template matching: si ruota un
ritaglio dell'immagine su un range di angoli candidati e si sceglie quello
che massimizza il match contro un template della freccia (fornito già
orientato correttamente: freccia orizzontale, puntata a sinistra).

OTTIMIZZAZIONE (necessaria: la ricerca naive su tutta l'immagine impiega
~16s/frame, troppo per una linea a 1 bottiglia/secondo con budget totale
<500ms includendo l'inferenza PatchCore):

  1. CROP RISTRETTO: il lavoro non avviene sull'intero tappo, ma su un
     ritaglio quadrato limitato alla sola fascia anulare che può contenere
     la freccia a qualunque angolo di rotazione (raggio della freccia dal
     centro + metà diagonale del template + margine) — non l'intero
     raggio del tappo.

  2. RICERCA A DUE STADI (coarse-to-fine):
       - Stage 1 (coarse): ricerca grossolana su tutti i 360°, a step
         largo (default 10°), su immagine ridotta di scala (default 20%).
         Individua rapidamente la zona angolare corretta e, per il match
         migliore, anche la POSIZIONE in cui la freccia è stata trovata
         nel crop.
       - Stage 2 (fine): raffinamento in un intorno ristretto del miglior
         angolo coarse (default ±6°, step 1°), MA con l'output di
         warpAffine limitato a una piccola finestra (template + margine)
         centrata sulla posizione stimata dallo stage 1 — non sull'intero
         crop. Il costo di warpAffine dipende dalla dimensione
         dell'OUTPUT, non dal crop sorgente: restringere l'output a una
         piccola finestra invece dell'intero crop ristretto porta il
         costo di ogni iterazione da centinaia di ms a pochi ms.

  3. La rotazione finale (l'angolo risultante dello stage 2) viene
     applicata UNA SOLA VOLTA, all'immagine a colori originale intera, per
     produrre l'output — stesso principio di align_cap_image.

Tempo misurato sulle immagini di riferimento: ~170ms totali (contro i
~16000ms dell'approccio naive), con angolo finale entro ~0.5° da quello
trovato con ricerca esaustiva a piena risoluzione, e score di match
altissimo (>0.95 su tappo integro, >0.85 su tappo con difetto vicino alla
zona della freccia).
"""

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Stima centro del tappo (Otsu, contorno più grande) — il tappo riempie
# quasi tutto il frame per questo tipo, a differenza dei tappi trasparenti
# che avevano molto sfondo nero attorno.
# --------------------------------------------------------------------------- #

def _estimate_cap_center_and_radius(gray: np.ndarray, soglia: int = 30) -> tuple[float, float, float] | None:
    """
    Stima centro e raggio esterno del tappo tramite soglia fissa +
    contorno più grande. Ritorna (cx, cy, raggio) o None se non trovato.
    """
    _, th = cv2.threshold(gray, soglia, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(biggest)
    return (float(cx), float(cy), float(r))


def _safe_crop(img: np.ndarray, cx: int, cy: int, half: int) -> tuple[np.ndarray, int, int]:
    """
    Ritaglia un quadrato di lato 2*half centrato su (cx, cy), gestendo i
    bordi dell'immagine con padding (replicate) se il centro è vicino al
    margine. Stessa utility già usata in alignment.py/alignment_small.py.
    """
    h, w = img.shape[:2]

    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)

    crop = img[y1c:y2c, x1c:x2c]

    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REPLICATE,
        )

    cx_local = cx - x1
    cy_local = cy - y1
    return crop, cx_local, cy_local


def find_cap_center_tappo4(frame_bgr: np.ndarray, soglia: int = 30) -> tuple[float, float] | None:
    """
    Stima il centro del tappo tipo 4 (per uso da augment_dataset.py, stesso
    ruolo di find_cap_center_via_markers per i tappi trasparenti — qui però
    basato su Otsu/soglia fissa + contorno più grande, non su marker Hough,
    perché questo tappo non ne ha di utilizzabili).
    """
    if frame_bgr is None:
        return None
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    result = _estimate_cap_center_and_radius(gray, soglia=soglia)
    if result is None:
        return None
    cx, cy, _ = result
    return (cx, cy)


# --------------------------------------------------------------------------- #
# Ricerca angolare coarse-to-fine (fine stage con output ristretto)
# --------------------------------------------------------------------------- #

def _best_angle_search(
    crop_gray: np.ndarray,
    template: np.ndarray,
    cx_local: float,
    cy_local: float,
    coarse_scale: float = 0.2,
    coarse_step_deg: int = 10,
    fine_range_deg: float = 6.0,
    fine_step_deg: float = 1.0,
    fine_window_margin: int = 60,
) -> tuple[float, float]:
    """
    Trova l'angolo di rotazione che massimizza il match del template
    contro crop_gray, con ricerca in due stadi:

      Stage 1 (coarse): su immagine ridotta di scala, step largo, su
      tutti i 360°. Oltre all'angolo migliore, tiene anche la posizione
      del match (serve per centrare la finestra dello stage 2).

      Stage 2 (fine): raffinamento in un intorno ristretto del miglior
      angolo coarse. warpAffine calcola SOLO una piccola finestra di
      output (template + margine) centrata sulla posizione stimata dallo
      stage 1, invece dell'intero crop — il costo di warpAffine dipende
      dalla dimensione dell'output, quindi questo riduce drasticamente il
      tempo per iterazione.

    Ritorna (angolo_gradi, score_match).
    """
    side = crop_gray.shape[0]

    # --- Stage 1: coarse, su immagine ridotta di scala ---
    small = cv2.resize(crop_gray, (int(side * coarse_scale), int(side * coarse_scale)))
    template_small = cv2.resize(
        template, (int(template.shape[1] * coarse_scale), int(template.shape[0] * coarse_scale))
    )
    cxl_s, cyl_s = cx_local * coarse_scale, cy_local * coarse_scale
    side_s = small.shape[0]

    best_angle, best_score, best_loc_s = 0.0, -1.0, (0, 0)
    for angle in range(-180, 180, coarse_step_deg):
        M = cv2.getRotationMatrix2D((cxl_s, cyl_s), angle, 1.0)
        rotated = cv2.warpAffine(
            small, M, (side_s, side_s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        res = cv2.matchTemplate(rotated, template_small, cv2.TM_CCOEFF_NORMED)
        _, score, _, maxloc = cv2.minMaxLoc(res)
        if score > best_score:
            best_score, best_angle, best_loc_s = score, float(angle), maxloc

    # Posizione stimata (piena risoluzione) del match trovato dallo stage 1,
    # usata per centrare la finestra ristretta dello stage 2.
    win_cx = int(best_loc_s[0] / coarse_scale) + template.shape[1] // 2
    win_cy = int(best_loc_s[1] / coarse_scale) + template.shape[0] // 2
    win_w = template.shape[1] + fine_window_margin * 2
    win_h = template.shape[0] + fine_window_margin * 2
    ox = win_cx - win_w // 2
    oy = win_cy - win_h // 2

    # --- Stage 2: fine, output ristretto alla sola finestra piccola ---
    best_angle_fine, best_score_fine = best_angle, -1.0
    start = int(best_angle * 10) - int(fine_range_deg * 10)
    end = int(best_angle * 10) + int(fine_range_deg * 10) + 1
    step = max(1, int(fine_step_deg * 10))
    for angle10 in range(start, end, step):
        angle = angle10 / 10.0
        M = cv2.getRotationMatrix2D((cx_local, cy_local), angle, 1.0)
        # Trasla la matrice per far sì che warpAffine calcoli SOLO la
        # piccola finestra [ox:ox+win_w, oy:oy+win_h], non l'intero crop.
        M_win = M.copy()
        M_win[0, 2] -= ox
        M_win[1, 2] -= oy
        rotated_win = cv2.warpAffine(
            crop_gray, M_win, (win_w, win_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        res = cv2.matchTemplate(rotated_win, template, cv2.TM_CCOEFF_NORMED)
        score = cv2.minMaxLoc(res)[1]
        if score > best_score_fine:
            best_score_fine, best_angle_fine = score, angle

    return best_angle_fine, best_score_fine


# --------------------------------------------------------------------------- #
# Funzione principale di allineamento
# --------------------------------------------------------------------------- #

def align_cap_image_black(
    img: np.ndarray,
    template: np.ndarray,
    arrow_radius_hint: float = 405.0,
    soglia_centro: int = 30,
    min_score: float = 0.5,
) -> np.ndarray:
    """
    Ruota il tappo tipo 4 in modo che la freccia risulti orizzontale e
    puntata a sinistra, tramite ricerca angolare via template matching
    (coarse-to-fine, con output ristretto nello stage fine per le
    prestazioni — vedi _best_angle_search).

    Parameters
    ----------
    img : np.ndarray
        Immagine sorgente (a colori, BGR).
    template : np.ndarray
        Template della freccia, GIA' orientato correttamente (orizzontale,
        puntata a sinistra) — es. template_matching_black.bmp.
    arrow_radius_hint : float
        Distanza approssimativa (in pixel) della freccia dal centro del
        tappo, usata per dimensionare il crop di ricerca. Calibrata a
        405px sulle immagini di riferimento; aggiornare se cambia
        l'ottica/working distance della camera.
    soglia_centro : int
        Soglia di binarizzazione per isolare il tappo dallo sfondo (il
        tappo riempie quasi tutto il frame per questo tipo).
    min_score : float
        Score minimo di match sotto il quale la rotazione viene
        considerata inaffidabile e NON applicata (meglio non ruotare alla
        cieca che applicare un angolo sbagliato con sicurezza).

    Returns
    -------
    np.ndarray
        L'immagine ruotata. Se il centro non viene rilevato o il match è
        troppo debole, ritorna l'immagine originale invariata.
    """
    if img is None or template is None:
        return img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    centro_info = _estimate_cap_center_and_radius(gray, soglia=soglia_centro)
    if centro_info is None:
        print("[WARNING] tappo4: centro non rilevato. Rotazione saltata.")
        return img
    cx, cy, _radius_esterno = centro_info
    cx, cy = int(cx), int(cy)

    # Crop ristretto alla fascia che può contenere la freccia a qualunque
    # rotazione: raggio della freccia + metà diagonale del template +
    # margine di sicurezza. NON usa il raggio esterno del tappo (troppo
    # grande, motivo della lentezza dell'approccio naive iniziale).
    template_diag_half = int(np.hypot(*template.shape) / 2)
    half = int(arrow_radius_hint + template_diag_half + 40)

    crop_gray, cx_local, cy_local = _safe_crop(gray, cx, cy, half)

    best_angle, best_score = _best_angle_search(crop_gray, template, cx_local, cy_local)

    if best_score < min_score:
        # Match troppo debole per fidarsi: meglio non ruotare alla cieca.
        print(f"[WARNING] tappo4: match debole (score={best_score:.3f}). Rotazione saltata.")
        return img

    # Applica la rotazione trovata UNA SOLA VOLTA, all'immagine a colori
    # intera (stesso principio di align_cap_image: tutte le operazioni di
    # ricerca sono su crop/finestre ristrette, solo l'output finale è
    # full-size).
    height, width = img.shape[:2]
    M_final = cv2.getRotationMatrix2D((float(cx), float(cy)), best_angle, 1.0)
    result = cv2.warpAffine(
        img, M_final, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    return result


if __name__ == "__main__":
    import time

    img = cv2.imread("old_result\preprocess\FONTE\TAPPO4-OK_PC23030XS_LTRNHP210W20_ITA50-GC-10C_WD100MM.bmp")
    template = cv2.imread("old_result\preprocess/template_matching_black.bmp", cv2.IMREAD_GRAYSCALE)

    t0 = time.perf_counter()
    res = align_cap_image_black(img, template)
    t1 = time.perf_counter()
    print(f"Tempo allineamento: {(t1 - t0) * 1000:.0f} ms")

    cv2.imwrite("test_output.bmp", res)