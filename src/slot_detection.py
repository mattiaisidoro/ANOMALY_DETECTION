"""
slot_detection.py
===================
Versione "in-memory" del rilevamento fessura/rotazione validato su banco
(vedi script standalone allinea_tappo_v4.py per la versione con debug image
e batch da cartella). Qui la stessa logica è esposta come funzione pura,
senza I/O su disco, per l'uso nel loop real-time di main.py.

La selezione della fessura si basa su FORMA + POSIZIONE RELATIVA del blob
scuro (non solo area), perché la zona di ricerca contiene tipicamente
anche un blob molto più grande e irregolare (sfondo visibile attraverso
l'anello di sicurezza del tappo) che va scartato:
  - elongation (minAreaRect, invariante alla rotazione) in un range atteso
  - distanza dal centro / raggio del tappo in un range atteso
  - area in un range atteso
Vedi allinea_tappo_v4.py per i dettagli della calibrazione.
"""

import cv2
import numpy as np
import math


# Default: usati se il profilo del tappo non specifica un parametro in
# ROTATION_PARAMS. Tenerli allineati a quelli calibrati in allinea_tappo_v4.py.
DEFAULTS = {
    "TARGET_ANGLE": 180.0,
    "CAP_THRESHOLD": 10,
    "SLOT_VAL_MAX": 30,
    "SEARCH_RADIUS_FACTOR": 1.15,
    "SLOT_AREA_MIN": 30000,
    "SLOT_AREA_MAX": 140000,
    "SLOT_ELONG_MIN": 3.0,
    "SLOT_ELONG_MAX": 6.5,
    "SLOT_DIST_RATIO_MIN": 0.65,
    "SLOT_DIST_RATIO_MAX": 0.87,
    "MIN_DIST_RATIO": 0.15,  # distanza minima centroide/raggio per considerarlo affidabile
}


def _estrai_maschera_tappo(img_gray: np.ndarray, cap_threshold: int) -> np.ndarray:
    _, mask = cv2.threshold(img_gray, cap_threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _trova_tappo(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None
    c = max(contours, key=cv2.contourArea)
    (cX, cY), radius = cv2.minEnclosingCircle(c)
    return c, (int(cX), int(cY)), radius


def _trova_angolo_slot(img_gray: np.ndarray, contorno, centro: tuple, raggio: float, p: dict):
    h, w = img_gray.shape[:2]
    cX, cY = centro

    hull = cv2.convexHull(contorno)
    mask_hull = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_hull, [hull], 255)

    mask_circle = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_circle, (cX, cY), int(raggio * p["SEARCH_RADIUS_FACTOR"]), 255, -1)

    zona = cv2.bitwise_and(mask_hull, mask_circle)

    mask_scuro = cv2.inRange(img_gray, 0, p["SLOT_VAL_MAX"])
    mask_slot_all = cv2.bitwise_and(mask_scuro, zona)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_slot_all = cv2.morphologyEx(mask_slot_all, cv2.MORPH_OPEN, kernel_small)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_slot_all, connectivity=8
    )
    if n_labels <= 1:
        return None, None

    candidati = []
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 200:
            continue
        cx_s, cy_s = centroids[i]
        dist = math.hypot(cx_s - cX, cy_s - cY)
        dist_ratio = dist / raggio

        blob_mask = np.where(labels == i, 255, 0).astype(np.uint8)
        cnts, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        (rw, rh) = cv2.minAreaRect(max(cnts, key=cv2.contourArea))[1]
        lato_lungo, lato_corto = max(rw, rh), max(min(rw, rh), 1)
        elong = lato_lungo / lato_corto

        ok = (p["SLOT_AREA_MIN"] <= area <= p["SLOT_AREA_MAX"] and
              p["SLOT_ELONG_MIN"] <= elong <= p["SLOT_ELONG_MAX"] and
              p["SLOT_DIST_RATIO_MIN"] <= dist_ratio <= p["SLOT_DIST_RATIO_MAX"])
        if ok:
            candidati.append((i, area, dist, cx_s, cy_s))

    if not candidati:
        return None, None

    idx_blob, _, dist, cx_s, cy_s = max(candidati, key=lambda t: t[1])

    if dist < raggio * p["MIN_DIST_RATIO"]:
        return None, None

    angle_deg = math.degrees(math.atan2(cy_s - cY, cx_s - cX)) % 360
    return angle_deg, (cx_s, cy_s)


def allinea_tappo_frame(frame_bgr: np.ndarray, rotation_params: dict | None = None):
    """
    Rileva la fessura della linguetta e ruota il frame di conseguenza.

    Parameters
    ----------
    frame_bgr : np.ndarray
        Frame a colori (BGR) così come letto da cv2.imread. Il contenuto
        può essere di fatto in scala di grigi (3 canali identici): la
        rilevazione lavora comunque sul canale di grigio derivato.
    rotation_params : dict, opzionale
        Override dei parametri di DEFAULTS (vedi ROTATION_PARAMS nel
        profilo del tappo in tappi_profili.yaml). Chiavi non presenti
        usano il default.

    Returns
    -------
    (frame_allineato_bgr, angolo_rilevato_deg, ok) : tuple
        Se ok=False, la fessura non è stata rilevata e frame_allineato_bgr
        è semplicemente una copia del frame originale (nessuna rotazione
        applicata) — sta al chiamante decidere come gestire il caso
        (skip frame, log, alert, ecc.).
    """
    p = {**DEFAULTS, **(rotation_params or {})}

    img_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr

    mask_tappo = _estrai_maschera_tappo(img_gray, p["CAP_THRESHOLD"])
    contorno, centro, raggio = _trova_tappo(mask_tappo)
    if contorno is None:
        return frame_bgr.copy(), None, False

    angolo, _punto_slot = _trova_angolo_slot(img_gray, contorno, centro, raggio, p)
    if angolo is None:
        return frame_bgr.copy(), None, False

    correzione = (p["TARGET_ANGLE"] - angolo + 360) % 360
    if correzione > 180:
        correzione -= 360

    h, w = frame_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((float(centro[0]), float(centro[1])), -correzione, 1.0)
    frame_allineato = cv2.warpAffine(frame_bgr, M, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)

    return frame_allineato, angolo, True