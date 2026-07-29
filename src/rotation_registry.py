"""
rotation_registry.py
=====================
Punto unico di dispatch tra "nome metodo di rotazione" (stringa presente nel
profilo del tappo, es. "slot_detection" o "template_matching") e la funzione
Python che lo implementa.

Ogni funzione di rotazione registrata ha la STESSA firma:

    align(frame_bgr: np.ndarray, profile: dict) -> np.ndarray

dove `profile` è il dizionario caricato da ConfigDB.get_tappo_profile(),
arricchito con un campo "_assets" (asset pre-caricati una volta sola
all'avvio, es. l'immagine template per il template matching — vedi
load_assets_for_profile più sotto).

Per aggiungere un nuovo metodo di rotazione:
  1. Scrivi la funzione con la firma sopra (o un adapter che la rispetti).
  2. Se il metodo ha bisogno di asset da pre-caricare (immagini, modelli
     leggeri, ecc.), aggiungi un loader in ASSET_LOADERS.
  3. Aggiungi la voce in ROTATION_REGISTRY.
  4. Usa quella chiave in ROTATION_METHOD nel profilo del tappo
     (config/tappi_profili.yaml). main.py non va toccato.
"""

import cv2
import numpy as np

from src.alignment_small import align_cap_image_small
from src.alignment import align_cap_image
from src.slot_detection import allinea_tappo_frame


# ──────────────────────────────────────────────────────────────────────────
# ADAPTER: template matching (tappi trasparenti grandi)
# ──────────────────────────────────────────────────────────────────────────

def _load_assets_template_matching(profile: dict) -> dict:
    template_path = profile.get("TEMPLATE_PATH")
    if not template_path:
        raise ValueError(
            "[ERRORE CRITICO] ROTATION_METHOD='template_matching' richiede "
            "TEMPLATE_PATH nel profilo del tappo, ma non è presente."
        )
    template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template_img is None:
        raise ValueError(
            f"[ERRORE CRITICO] Template di rotazione non trovato in '{template_path}'."
        )
    return {"template_img": template_img}


def _rotate_template_matching(frame_bgr: np.ndarray, profile: dict) -> np.ndarray:
    template_img = profile["_assets"]["template_img"]
    return align_cap_image(frame_bgr, template_img)

# ──────────────────────────────────────────────────────────────────────────
# ADAPTER: template matching (tappi trasparenti piccoli)
# ──────────────────────────────────────────────────────────────────────────
def _load_assets_template_matching_small(profile: dict) -> dict:
    template_path = profile.get("TEMPLATE_PATH")
    if not template_path:
        raise ValueError(
            "[ERRORE CRITICO] ROTATION_METHOD='template_matching_small' richiede "
            "TEMPLATE_PATH nel profilo del tappo, ma non è presente."
        )
    template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template_img is None:
        raise ValueError(
            f"[ERRORE CRITICO] Template di rotazione non trovato in '{template_path}'."
        )
    return {"template_img": template_img}

def _rotate_template_matching_small(frame_bgr: np.ndarray, profile: dict) -> np.ndarray:
    template_img = profile["_assets"]["template_img"]
    return align_cap_image_small(frame_bgr, template_img)


# ──────────────────────────────────────────────────────────────────────────
# ADAPTER: slot detection su grigio (tappo rosso piccolo)
# ──────────────────────────────────────────────────────────────────────────

def _load_assets_slot_detection(profile: dict) -> dict:
    # Nessun asset da pre-caricare: tutti i parametri sono in ROTATION_PARAMS.
    return {}


def _rotate_slot_detection(frame_bgr: np.ndarray, profile: dict) -> np.ndarray:
    params = profile.get("ROTATION_PARAMS", {})
    aligned, angolo, ok = allinea_tappo_frame(frame_bgr, params)
    if not ok:
        print("[WARNING] slot_detection: fessura non rilevata, uso il frame non ruotato.")
        return frame_bgr
    return aligned




# ──────────────────────────────────────────────────────────────────────────
# REGISTRY
# ──────────────────────────────────────────────────────────────────────────
# Chiave = valore di ROTATION_METHOD nel profilo del tappo (tappi_profili.yaml)

ROTATION_REGISTRY = {
    "template_matching": _rotate_template_matching,
    "slot_detection": _rotate_slot_detection,
    "template_matching_small": _rotate_template_matching_small,
    # "pca_alignment": _rotate_pca,              # TODO quando pronto
    # "hough_alignment": _rotate_hough,           # TODO quando pronto
}

ASSET_LOADERS = {
    "template_matching": _load_assets_template_matching,
    "slot_detection": _load_assets_slot_detection,
    "template_matching_small": _load_assets_template_matching_small,
    # "pca_alignment": _load_assets_pca,
    # "hough_alignment": _load_assets_hough,
}


def get_rotation_fn(rotation_method: str):
    """
    Restituisce la funzione di rotazione registrata per `rotation_method`.
    Solleva ValueError con messaggio chiaro se il metodo non è registrato
    (fail-fast: meglio bloccarsi subito all'avvio che scoprirlo a runtime
    su una fessura non rilevata).
    """
    if rotation_method not in ROTATION_REGISTRY:
        disponibili = ", ".join(sorted(ROTATION_REGISTRY.keys()))
        raise ValueError(
            f"[ERRORE CRITICO] ROTATION_METHOD='{rotation_method}' non registrato "
            f"in rotation_registry.py. Metodi disponibili: {disponibili}"
        )
    return ROTATION_REGISTRY[rotation_method]


def load_assets_for_profile(profile: dict) -> dict:
    """
    Pre-carica gli asset necessari al metodo di rotazione del profilo
    (es. l'immagine template), una sola volta all'avvio. Ritorna un dict
    vuoto se il metodo non ha asset da caricare.
    """
    rotation_method = profile["ROTATION_METHOD"]
    if rotation_method not in ASSET_LOADERS:
        disponibili = ", ".join(sorted(ASSET_LOADERS.keys()))
        raise ValueError(
            f"[ERRORE CRITICO] ROTATION_METHOD='{rotation_method}' non ha un "
            f"loader di asset registrato in rotation_registry.py. "
            f"Metodi disponibili: {disponibili}"
        )
    return ASSET_LOADERS[rotation_method](profile)