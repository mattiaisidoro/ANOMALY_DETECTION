"""
augment_dataset.py
====================
Genera un dataset sintetico ("farlocco") di immagini GOOD per il training
one-class di PatchCore, a partire da poche foto reali di tappi buoni.

IDEA
----
PatchCore (come tutti i modelli one-class per anomaly detection) impara
la "normalità" osservando solo esempi GOOD — non servono label di difetto.
Con pochissime foto sorgente il training sarebbe troppo povero: qui si
generano molte varianti di ciascuna foto ruotandola di un piccolo angolo
casuale ATTORNO AL CENTRO GEOMETRICO REALE DEL TAPPO. Ruotare attorno al
centro del tappo, non al centro dell'immagine, è essenziale: altrimenti
il tappo si sposterebbe lateralmente e uscirebbe dalla ROI configurata
per l'anomaly detection.

Il centro del tappo viene stimato come punto medio tra i due marker fisici
rilevati via Hough (stessa logica robusta già usata per l'allineamento in
produzione dei tappi trasparenti — vedi src/alignment.py e
src/alignment_small.py, funzione find_cap_center_via_markers), NON con la
soglia fissa di src/slot_detection.py: quella è calibrata per il tappo
rosso opaco e non è adatta a tappi trasparenti (dove una soglia fissa non
produce una maschera pulita).

Il senso pratico: sulla linea reale, anche dopo l'allineamento automatico,
il tappo non sarà MAI perfettamente a 180° esatti — ci sarà sempre un piccolo
errore residuo (qualche grado). Allenare PatchCore anche su piccole
rotazioni intorno al target aiuta il modello a non segnalare come "anomalia"
quel residuo fisiologico di errore di allineamento.

⚠️ LIMITE IMPORTANTE (da tenere a mente):
Questo è un trucco di BOOTSTRAP, non sostituisce foto reali. La rotazione
non introduce alcuna variabilità su illuminazione, sporco, usura, piccole
imperfezioni di stampaggio reali: se le foto sorgente condividono uno
sfondo pulito e identico, il modello imparerà SOLO quella variabilità
angolare, non la variabilità reale della linea. Appena hai foto vere della
linea, ri-allena (o quantomeno arricchisci il dataset) con quelle.

⚠️ ATTENZIONE AGLI ARTEFATTI DI BORDO:
cv2.warpAffine con BORDER_REPLICATE, su rotazioni anche piccole, può
introdurre pattern ai bordi/angoli dell'immagine che non esistono nelle
foto reali della linea (pixel "stirati" dalla replicazione). Se questi
pattern ricadono dentro la ROI configurata per il tappo, PatchCore può
imparare a considerarli "normali" (rischio overfitting su artefatto) o,
peggio, il modello di produzione può vederli come fortemente anomali se
non presenti in training in tutte le angolazioni. Tenere d'occhio
soprattutto se il punteggio di anomalia in test satura sempre al massimo
possibile: è un sintomo tipico di questo problema (vedi nota in fondo
al file).

USO
---
    python augment_dataset.py \
        --input cartella_foto_sorgenti \
        --output cartella_dataset_generato \
        --target 120 \
        --angle-max 2.5 \
        --seed 42

Ogni immagine sorgente contribuisce con circa target/N_sorgenti varianti
(rotazioni casuali uniformi in [-angle-max, +angle-max] gradi, esclusi gli
angoli troppo vicini a 0 se --min-angle è specificato), più l'originale
stesso (rotazione 0°) incluso una volta per sorgente.
"""

import argparse
import os
import random
import sys

import cv2
import numpy as np

# Risale automaticamente le cartelle a partire dalla posizione di questo
# script, finché non trova una cartella che contiene "src/alignment_small.py"
# (cioè la root del progetto principale), e la aggiunge a sys.path.
# Questo permette di eseguire augment_dataset.py da QUALUNQUE sottocartella
# (es. old_result/preprocess/) senza dover copiare file o hardcodare path
# assoluti che si romperebbero spostando lo script.
def _trova_e_registra_project_root() -> None:
    corrente = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):  # limite di sicurezza, evita loop infiniti
        candidato = os.path.join(corrente, "src", "alignment_small.py")
        if os.path.exists(candidato):
            if corrente not in sys.path:
                sys.path.insert(0, corrente)
            return
        parent = os.path.dirname(corrente)
        if parent == corrente:  # arrivati alla radice del filesystem
            break
        corrente = parent
    print(
        "[WARNING] Non ho trovato 'src/alignment_small.py' risalendo le cartelle "
        f"da '{os.path.dirname(os.path.abspath(__file__))}'. "
        "L'import di find_cap_center_via_markers potrebbe fallire."
    )


_trova_e_registra_project_root()

try:
    from src.alignment_small import find_cap_center_via_markers
except ImportError:
    from alignment_small import find_cap_center_via_markers


def ruota_attorno_al_tappo(frame_bgr: np.ndarray, centro: tuple, angolo_deg: float) -> np.ndarray:
    """Ruota il frame di `angolo_deg` gradi attorno al centro del tappo (non dell'immagine)."""
    h, w = frame_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((float(centro[0]), float(centro[1])), angolo_deg, 1.0)
    return cv2.warpAffine(frame_bgr, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def genera_dataset(input_dir: str, output_dir: str, target_count: int,
                   angle_max: float, min_angle: float, seed: int) -> None:
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    estensioni = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    sorgenti = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(estensioni))

    if not sorgenti:
        print(f"[ERRORE] Nessuna immagine trovata in '{input_dir}'.")
        return

    n_sorgenti = len(sorgenti)
    # Ogni sorgente contribuisce con lo stesso numero di varianti (+1 per l'originale a 0°)
    varianti_per_sorgente = max(1, (target_count // n_sorgenti) - 1)

    print(f"[INFO] {n_sorgenti} immagini sorgente trovate in '{input_dir}'.")
    print(f"[INFO] Genero {varianti_per_sorgente} varianti ruotate + l'originale per ciascuna "
          f"(totale atteso: ~{n_sorgenti * (varianti_per_sorgente + 1)} immagini).")

    totale_generate = 0
    totale_fallite = 0

    for nome_file in sorgenti:
        path = os.path.join(input_dir, nome_file)
        frame = cv2.imread(path)
        if frame is None:
            print(f"[WARNING] Impossibile leggere '{nome_file}', salto.")
            continue

        centro = find_cap_center_via_markers(frame)
        if centro is None:
            print(f"[WARNING] Marker non rilevati in '{nome_file}', salto (nessuna variante generata).")
            totale_fallite += 1
            continue

        nome_base, ext = os.path.splitext(nome_file)

        # 1) L'originale (rotazione 0°), per non "diluire" via interpolazione l'unica foto reale
        out_path = os.path.join(output_dir, f"{nome_base}_orig{ext}")
        cv2.imwrite(out_path, frame)
        totale_generate += 1

        # 2) Le varianti ruotate di un piccolo angolo casuale attorno al centro reale del tappo
        for i in range(varianti_per_sorgente):
            # Angolo casuale in [-angle_max, angle_max], escludendo la fascia [-min_angle, min_angle]
            # per evitare varianti troppo simili all'originale se min_angle > 0.
            if min_angle > 0:
                segno = random.choice([-1, 1])
                angolo = segno * random.uniform(min_angle, angle_max)
            else:
                angolo = random.uniform(-angle_max, angle_max)

            ruotata = ruota_attorno_al_tappo(frame, centro, angolo)
            out_path = os.path.join(output_dir, f"{nome_base}_rot{i:03d}_{angolo:+.1f}deg{ext}")
            cv2.imwrite(out_path, ruotata)
            totale_generate += 1

    print(f"\n[COMPLETATO] {totale_generate} immagini generate in '{output_dir}' "
          f"({totale_fallite} sorgenti scartate per marker non rilevati).")
    if totale_generate < target_count:
        print(f"[NOTA] Il totale generato ({totale_generate}) è sotto il target richiesto "
              f"({target_count}): aumenta --target o aggiungi più immagini sorgente.")


def main():
    parser = argparse.ArgumentParser(
        description="Genera un dataset sintetico di tappi trasparenti GOOD tramite piccole rotazioni "
                    "attorno al centro reale del tappo (stimato via coppia di marker), per il "
                    "bootstrap di PatchCore."
    )
    parser.add_argument("--input", required=True, help="Cartella con le foto sorgente (poche, es. 3)")
    parser.add_argument("--output", required=True, help="Cartella dove salvare il dataset generato")
    parser.add_argument("--target", type=int, default=120, help="Numero totale di immagini desiderato (default: 120)")
    parser.add_argument("--angle-max", type=float, default=2.5,
                        help="Massimo angolo di rotazione in gradi, in ogni direzione (default: 2.5)")
    parser.add_argument("--min-angle", type=float, default=0.5,
                        help="Angolo minimo in gradi (evita varianti troppo vicine a 0°, default: 0.5)")
    parser.add_argument("--seed", type=int, default=42, help="Seed random per riproducibilità")
    args = parser.parse_args()

    genera_dataset(
        input_dir=args.input,
        output_dir=args.output,
        target_count=args.target,
        angle_max=args.angle_max,
        min_angle=args.min_angle,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()


    