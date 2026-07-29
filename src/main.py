"""
main.py
================
Real-time AI-powered visual inspection pipeline using PatchCore anomaly detection.

OVERVIEW
--------
This module implements a production-ready inspection loop that:
  1. Watches a buffer directory for incoming PNG/BMP frames.
  2. Aligns each frame using the rotation method configured for the ACTIVE
     cap type (see TAPPO_TYPE below).
  3. Runs anomaly detection via either a TorchInferencer or OpenVINOInferencer
     (auto-selected based on the model path), using the model configured
     for the active cap type.
  4. Computes a region-of-interest anomaly score on the cap area (ROI also
     configured per cap type).
  5. Classifies the frame as GOOD or REJECT and saves an annotated output image.

CAP TYPE CONFIGURATION
-----------------------
Different physical cap types require different anomaly detection models,
different regions of interest, and different rotation/alignment algorithms
(e.g. slot detection on grayscale vs. template matching). Rather than
branching on cap type throughout this file, all cap-specific configuration
lives in ``config/tappi_profili.yaml``, keyed by a profile name. The DB
parameter ``TAPPO_TYPE`` (see config_base.yaml / ConfigDB) selects which
profile is active. The profile is read ONCE at startup — changing
TAPPO_TYPE requires restarting the process (this is intentional: the
process is expected to be restarted at every batch/line change).

Adding a new cap type: add a profile block to tappi_profili.yaml (model
path, ROI, ROTATION_METHOD, ROTATION_PARAMS, threshold). If it needs a new
rotation algorithm, add it to src/rotation_registry.py. main.py itself
does not need to change.

CONFIGURATION
-------------
Global parameters (valid regardless of cap type) are read from the DB via
``config.db_manager.ConfigDB.get_param()``:

    BUFFER_DIR          str   Directory polled for incoming frames
    OUTPUT_DIR          str   Directory where result images are written
    DEVICE              str   Default compute device: "cpu", "cuda", or "cuda:<index>"
    TAPPO_TYPE           str   Active cap type; selects the profile below
    PAUSA_BUFFER_VUOTO  float Sleep interval (s) when the buffer is empty
    OVERLAY_ORIGINALE   float Blend weight for the original frame in the overlay
    OVERLAY_HEATMAP      float Blend weight for the heatmap in the overlay

Per-cap-type parameters are read from ``config/tappi_profili.yaml`` via
``ConfigDB.get_tappo_profile(tappo_type)``:

    MODEL_PATH           str   Path to the model file (.pt for Torch, .xml for OpenVINO)
    DEVICE                str   Overrides the global DEVICE, if set
    ROTATION_METHOD       str   Key into src/rotation_registry.py ROTATION_REGISTRY
    ROTATION_PARAMS       dict  Parameters passed to the rotation function
    TAPPO_Y1/Y2/X1/X2    int   Pixel bounding box of the cap ROI inside the full frame
    SOGLIA_TAPPO          float Anomaly score threshold for rejection

USAGE
-----
    python main.py

DEPENDENCIES
------------
    opencv-python, numpy, torch, anomalib, openvino
"""

import logging
import os
import time
from logging.handlers import RotatingFileHandler
from typing import Any

import cv2
import numpy as np
import torch

os.environ["TRUST_REMOTE_CODE"] = "1"

from anomalib.deploy import OpenVINOInferencer, TorchInferencer

from config.db_manager import ConfigDB

from src.rotation_registry import get_rotation_fn, load_assets_for_profile


# --------------------------------------------------------------------------- #
# Policy di salvataggio output
# --------------------------------------------------------------------------- #
# I frame REJECT vengono SEMPRE salvati (servono per tarare la soglia,
# rispondere a contestazioni, capire perché un pezzo è stato scartato).
#
# I frame GOOD, in produzione, in genere non servono a nessuno e sono la
# stragrande maggioranza dei frame (es. ~170k/giorno a 2 fps) -> saturano
# il disco in fretta se salvati tutti sempre.
#
# Default di PRODUZIONE: salva solo i REJECT.
SALVA_ANCHE_GOOD = False

# Per DEMO/COLLAUDO: decommenta la riga sotto per salvare anche i GOOD.
# Per tornare alla policy di produzione (solo REJECT), ricommentala: il
# default sopra (False) torna in vigore da solo, senza bisogno di
# modificare nient'altro.
SALVA_ANCHE_GOOD = True


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
# Sostituisce i print() sparsi nel file con il modulo logging standard:
#   - timestamp automatico su ogni riga
#   - livelli di gravità reali (DEBUG/INFO/WARNING/ERROR), non solo
#     prefissi testuali scritti a mano
#   - file di log con rotazione (non cresce all'infinito come farebbe
#     un semplice reindirizzamento dei print su file)
#   - la console continua a mostrare le informazioni importanti
#     (parametri caricati, profilo attivo, verdetto di ogni immagine),
#     mentre il file su disco cattura anche il dettaglio DEBUG per il
#     troubleshooting.
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "inspection.log")
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
LOG_BACKUP_COUNT = 5  # tiene le ultime 5 rotazioni -> max ~50 MB totali

logger = logging.getLogger("inspection")


def setup_logging() -> None:
    """
    Configura il logger di modulo: file con rotazione (DEBUG e superiori)
    + console (INFO e superiori). Va chiamata una sola volta, all'avvio
    di main().
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# --------------------------------------------------------------------------- #
# Device resolution
# --------------------------------------------------------------------------- #

def resolve_device(device_str: str) -> str:
    """
    Resolve a device string, falling back to CPU if CUDA is unavailable.
    """
    device_str = device_str.strip().lower()

    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"

        if ":" in device_str:
            try:
                idx = int(device_str.split(":")[1])
                if idx < torch.cuda.device_count():
                    return device_str
                logger.warning(f"GPU {idx} not found. Falling back to CPU.")
                return "cpu"
            except ValueError:
                pass

        return "cuda"

    return "cpu"


# --------------------------------------------------------------------------- #
# Pre-processing
# --------------------------------------------------------------------------- #

def preprocess(img: np.ndarray) -> np.ndarray | bool:
    """
    Pre-process an input frame for anomaly inference.
    """
    if img is None:
        logger.debug("Cannot open image.")
        return False

    gray: np.ndarray = (
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    )

    blurred: np.ndarray = cv2.GaussianBlur(gray, (11, 11), 0)

    return blurred


# --------------------------------------------------------------------------- #
# Heatmap utilities
# --------------------------------------------------------------------------- #

def normalize_heatmap(raw: np.ndarray) -> np.ndarray:
    """
    Squeeze and min-max normalise a raw anomaly map to [0, 1].
    """
    raw_2d: np.ndarray = np.squeeze(raw)
    h_min: float = float(raw_2d.min())
    h_max: float = float(raw_2d.max())
    if h_max > h_min:
        return (raw_2d - h_min) / (h_max - h_min)
    return raw_2d - h_min


def compute_cap_score(heatmap_full: np.ndarray, cfg: dict[str, Any]) -> float:
    """
    Compute the anomaly score for the cap region of interest.

    The percentile used is PERCENTILE_TAPPO from the active cap-type
    profile (different cap types/ROI sizes can need a different
    percentile to get a robust, non-noise-sensitive score).
    `cfg` here is the RUNTIME config for the active cap type (global params
    merged with the active profile — see build_runtime_cfg), so TAPPO_Y1..X2
    reflect the ROI of the currently selected cap type.
    """
    H, W = heatmap_full.shape

    y1: int = max(0, int(cfg.get("TAPPO_Y1", 200)))
    y2: int = min(H, int(cfg.get("TAPPO_Y2", 310)))
    x1: int = max(0, int(cfg.get("TAPPO_X1", 100)))
    x2: int = min(W, int(cfg.get("TAPPO_X2", 380)))

    percentile: float = float(cfg.get("PERCENTILE_TAPPO", 80))
   #DEBUG
   # print(f"compute_cap_score: ROI=({x1},{y1})-({x2},{y2}), percentile={percentile}")
    
    
    cap_roi: np.ndarray = heatmap_full[y1:y2, x1:x2]
    #debug
    #print("min/max heatmap_raw_full nella ROI:", cap_roi.min(), cap_roi.max())
    #print("percentili nella ROI:", np.percentile(cap_roi, [5, 15, 50, 80, 95, 99]))

    return float(np.percentile(cap_roi, percentile))


def create_overlay(frame_bgr: np.ndarray, heatmap_norm: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """
    Blend a colour heatmap over the original frame.
    """
    h, w = frame_bgr.shape[:2]
    heatmap_8bit: np.ndarray = (heatmap_norm * 255).astype(np.uint8)
    heatmap_color: np.ndarray = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
    heatmap_color = cv2.resize(heatmap_color, (w, h))
    return cv2.addWeighted(
        frame_bgr, cfg.get("OVERLAY_ORIGINALE", 0.7),
        heatmap_color, cfg.get("OVERLAY_HEATMAP", 0.3),
        0,
    )


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #

def classify(cap_score: float, cfg: dict[str, Any]) -> tuple[str, str]:
    """
    Classify a frame as GOOD or REJECT based on its cap anomaly score.
    """
    threshold: float = cfg.get("SOGLIA_TAPPO", 0.65)
    if cap_score >= threshold:
        return "REJECT", "CAP_DEFECT"
    return "GOOD", ""


# --------------------------------------------------------------------------- #
# Model backend resolution
# --------------------------------------------------------------------------- #

def resolve_model_backend(model_path: str) -> str:
    """
    Determina UNA SOLA VOLTA il backend del modello a partire dalla sua
    estensione, invece di due controlli-stringa indipendenti e non
    complementari (bug H2 nella review: un path come
    '.../openvino/model.xml' sceglieva OpenVINO in load_model() ma poi
    falliva il check separato "openvino_" in model_path.lower() nel loop,
    causando un preprocessing del tensore sbagliato senza nessun errore
    visibile).

    Ritorna "torch" o "openvino". Solleva ValueError per estensioni non
    riconosciute, per fallire subito invece di indovinare.
    """
    ext = os.path.splitext(model_path)[1].lower()
    if ext in (".pt", ".pth"):
        return "torch"
    if ext == ".xml":
        return "openvino"
    raise ValueError(
        f"[ERRORE CRITICO] Impossibile determinare il backend per MODEL_PATH='{model_path}' "
        f"(estensione '{ext}' non riconosciuta: attese .pt/.pth per Torch o .xml per OpenVINO)."
    )


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_model(
    model_path: str,
    device_str: str,
    backend: str,
) -> TorchInferencer | OpenVINOInferencer:
    """
    Instantiate the appropriate inferencer based on the resolved backend
    (see resolve_model_backend — il path stesso non viene più ispezionato
    qui, per avere un'unica fonte di verità condivisa col preprocessing).
    """
    device: str = resolve_device(device_str)

    logger.debug(f"Model path      : {model_path}")
    logger.debug(f"Model backend   : {backend}")
    logger.debug(f"Requested device: '{device_str}' -> resolved to: '{device}'")
    logger.debug(f"PyTorch CUDA    : {torch.cuda.is_available()}")

    if backend == "torch":
        return TorchInferencer(path=model_path, device=device)

    openvino_device: str = "GPU" if device.lower().startswith("cuda") else "CPU"
    return OpenVINOInferencer(model_path, openvino_device)


# --------------------------------------------------------------------------- #
# Cap-type setup
# --------------------------------------------------------------------------- #

def build_runtime_cfg(global_cfg: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """
    Merge global config and the active cap-type profile into a single dict
    used throughout the loop (ROI, threshold, overlay weights, ecc.).
    Profile keys take precedence over global ones on overlap (currently
    only DEVICE can overlap, and only if the profile sets it explicitly).
    """
    runtime = dict(global_cfg)
    for k, v in profile.items():
        if k.startswith("_"):
            continue  # campi interni (es. _assets), non fanno parte del runtime cfg
        if k == "DEVICE" and v is None:
            continue  # profilo non specifica DEVICE -> mantieni quello globale
        runtime[k] = v
    return runtime


# --------------------------------------------------------------------------- #
# Per-frame processing (estratto per poter essere avvolto in try/except)
# --------------------------------------------------------------------------- #

def process_frame(
    frame_bgr: np.ndarray,
    filename: str,
    profile: dict[str, Any],
    runtime_cfg: dict[str, Any],
    rotation_fn,
    inference: "TorchInferencer | OpenVINOInferencer",
    model_backend: str,
) -> tuple[str, str, float, float]:
    """
    Elabora un singolo frame già letto da disco: allineamento, inferenza,
    scoring, classificazione e salvataggio dell'immagine annotata.

    Ritorna (verdict, reason, cap_score, latency_ms).

    Qualsiasi eccezione sollevata qui (rotazione, predict, resize, slicing
    ROI, imwrite, ecc.) si propaga al chiamante: main() la cattura per
    quarantenare il frame senza fermare la linea (vedi C1 nella review).
    """
    t_start: float = time.perf_counter()

    # ALIGNMENT PHASE
    # Rotate/warp the image using the method configured for the active
    # cap type, so it comes out with the correct orientation.
    frame_bgr = rotation_fn(frame_bgr, profile)
    t_fine_rot: float = time.perf_counter()
    t_rot = (t_fine_rot - t_start) * 1000

    frame_rgb: np.ndarray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Build the input tensor — shape depends on model variant.
    # NOTA (fix H2): usa lo stesso `model_backend` risolto una sola volta
    # da resolve_model_backend() e già usato per scegliere l'inferencer in
    # load_model(), invece di un secondo check indipendente sul path.
    if model_backend == "openvino":
        frame_resized: np.ndarray = cv2.resize(frame_rgb, (256, 256))
        frame_chw: np.ndarray = np.transpose(frame_resized, (2, 0, 1))
        input_tensor: np.ndarray = np.expand_dims(frame_chw.astype(np.float32) / 255.0, 0)
    else:
        input_tensor = frame_rgb

    prediction = inference.predict(image=input_tensor)
    
    # DEBUG diagnostico
    """print("pred_score:", getattr(prediction, "pred_score", "non presente"))
    print("pred_label:", getattr(prediction, "pred_label", "non presente"))"""
    raw_before_squeeze = prediction.anomaly_map
    if hasattr(raw_before_squeeze, "detach"):
        raw_before_squeeze = raw_before_squeeze.detach().cpu().numpy()
    #print("min/max anomaly_map GREZZA (256x256, pre-resize, pre-squeeze):", raw_before_squeeze.min(), raw_before_squeeze.max())

    anomaly_map: np.ndarray = prediction.anomaly_map
    if hasattr(anomaly_map, "detach"):
        anomaly_map = anomaly_map.detach().cpu().numpy()
    if isinstance(anomaly_map, dict):
        anomaly_map = anomaly_map.get("anomaly_map", anomaly_map)

    heatmap_raw: np.ndarray = np.squeeze(anomaly_map)  # NEW
    heatmap_norm: np.ndarray = normalize_heatmap(anomaly_map)

    heatmap_raw_full: np.ndarray = cv2.resize(heatmap_raw, (frame_bgr.shape[1], frame_bgr.shape[0]))

    heatmap_full: np.ndarray = cv2.resize(
        heatmap_norm, (frame_bgr.shape[1], frame_bgr.shape[0])
    )
    """if np.allclose(heatmap_raw_full, heatmap_full, atol=1e-6):
        print("Heatmap uguali")
    else:
        print("Heatmap diverse")
        
      #DEBUG
    print("shape anomaly_map grezza (prima del resize):", anomaly_map.shape)
    print("shape frame:", frame_bgr.shape)
    print("shape heatmap_raw_full (dopo resize):", heatmap_raw_full.shape)"""
#--------------------------------------
    cap_score: float = compute_cap_score(heatmap_raw_full, runtime_cfg)  # HEATMAP_RAW_FULL prima

    verdict, reason = classify(cap_score, runtime_cfg)
    latency_ms: float = (time.perf_counter() - t_start) * 1000

    # Build and save annotated output (heatmap overlay on cap ROI only).
    # Policy di retention: i REJECT si salvano sempre; i GOOD solo se
    # SALVA_ANCHE_GOOD è attivo (vedi flag a inizio file) — altrimenti in
    # produzione si accumulerebbero centinaia di migliaia di immagini al
    # giorno senza reale utilità, saturando il disco.
    deve_salvare: bool = (verdict == "REJECT") or SALVA_ANCHE_GOOD

    if deve_salvare:
        prefix: str = f"REJECT_{reason}" if verdict == "REJECT" else "GOOD"
        output_path: str = os.path.join(
            runtime_cfg.get("OUTPUT_DIR"),
            f"{prefix}_{filename.replace('.png', '.bmp')}",
        )

        H, W = frame_bgr.shape[:2]
        y1: int = max(0, int(runtime_cfg.get("TAPPO_Y1", 200)))
        y2: int = min(H, int(runtime_cfg.get("TAPPO_Y2", 310)))
        x1: int = max(0, int(runtime_cfg.get("TAPPO_X1", 100)))
        x2: int = min(W, int(runtime_cfg.get("TAPPO_X2", 380)))

        cap_bgr: np.ndarray = frame_bgr[y1:y2, x1:x2]
        cap_heatmap: np.ndarray = heatmap_full[y1:y2, x1:x2]
        cap_overlay: np.ndarray = create_overlay(cap_bgr, cap_heatmap, runtime_cfg)

        output_frame: np.ndarray = frame_bgr.copy()
        output_frame[y1:y2, x1:x2] = cap_overlay

        # Fix H4: cv2.imwrite ritorna False in caso di errore (es. disco pieno,
        # permessi, path non valido) invece di sollevare un'eccezione. Se il
        # valore non viene controllato, il chiamante crede che il salvataggio
        # sia andato a buon fine e cancella comunque il frame di input subito
        # dopo, perdendo sia l'output che l'input (nessuna traccia del verdetto).
        scrittura_ok: bool = cv2.imwrite(output_path, output_frame)
        if not scrittura_ok:
            raise IOError(f"cv2.imwrite ha fallito il salvataggio di '{output_path}'")
    else:
        # GOOD non salvato per policy di retention (SALVA_ANCHE_GOOD=False).
        # Nessun file scritto: nessun disco consumato per questo frame.
        pass

    logger.info(
        f"{filename} -> {verdict} {reason} | "
        f"cap score: {cap_score:.3f} | latency: {latency_ms:.1f} ms"
        #f"cap score: {cap_score:.3f} | latency: {latency_ms:.1f} ms rot takes {t_rot}ms"
    
    )

    return verdict, reason, cap_score, latency_ms


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def main() -> None:
    """
    Entry point: load the active cap-type profile, model and rotation
    method, then run the real-time inspection loop.
    """
    setup_logging()
    logger.info("STARTING AI INSPECTION MODEL — REAL TIME")

    db = ConfigDB()
    cfg: dict[str, Any] = db.get_param()
    logger.debug(f"Global parameters loaded from DB: {cfg}")

    tappo_type: str = cfg["TAPPO_TYPE"]
    logger.info(f"Active cap type (TAPPO_TYPE): '{tappo_type}'")

    profile: dict[str, Any] = db.get_tappo_profile(tappo_type)
    logger.debug(f"Profile loaded for '{tappo_type}': {profile}")

    rotation_fn = get_rotation_fn(profile["ROTATION_METHOD"])
    profile["_assets"] = load_assets_for_profile(profile)
    logger.info(f"Rotation method '{profile['ROTATION_METHOD']}' ready.")

    runtime_cfg: dict[str, Any] = build_runtime_cfg(cfg, profile)

    os.makedirs(runtime_cfg.get("BUFFER_DIR", "buffer_telecamera"), exist_ok=True)
    os.makedirs(runtime_cfg.get("OUTPUT_DIR", "risultati_linea"), exist_ok=True)

    # Cartella di quarantena: qui finiscono i frame che hanno causato
    # un'eccezione durante l'elaborazione, così non vengono né persi né
    # ritentati all'infinito (vedi C1 nella review).
    quarantine_dir: str = os.path.join(runtime_cfg.get("BUFFER_DIR", "buffer_telecamera"), "quarantena")
    os.makedirs(quarantine_dir, exist_ok=True)

    model_path: str | None = profile.get("MODEL_PATH")
    if not model_path or not os.path.exists(model_path):
        logger.error(f"Model not found or not set: {model_path}")
        return

    device_str: str = profile.get("DEVICE") or runtime_cfg.get("DEVICE", "cpu")

    # Fix H2: il backend viene risolto UNA VOLTA SOLA qui, e riusato sia per
    # istanziare l'inferencer sia per il preprocessing del tensore in
    # process_frame, invece di due controlli-stringa indipendenti che
    # potevano disallinearsi silenziosamente.
    model_backend: str = resolve_model_backend(model_path)

    logger.info(f"Loading model from {model_path}")
    t_load_start: float = time.perf_counter()
    inference: TorchInferencer | OpenVINOInferencer = load_model(
        model_path=model_path, device_str=device_str, backend=model_backend
    )
    t_load_ms: float = (time.perf_counter() - t_load_start) * 1000
    logger.info(f"PATCHCORE MODEL READY IN {t_load_ms:.2f} ms — WAITING FOR IMAGES")

    total_latency_ms: float = 0.0
    frame_count: int = 0

    while True:
        pending_files: list[str] = sorted(
            f for f in os.listdir(runtime_cfg.get("BUFFER_DIR")) if f.endswith(".bmp")
        )

        if not pending_files:
            time.sleep(runtime_cfg.get("PAUSA_BUFFER_VUOTO", 0.05))
            continue

        filename: str = pending_files[0]
        frame_path: str = os.path.join(runtime_cfg.get("BUFFER_DIR"), filename)

        frame_bgr: np.ndarray | None = cv2.imread(frame_path)
        if frame_bgr is None:
            time.sleep(0.01)
            continue

        try:
            verdict, reason, cap_score, latency_ms = process_frame(
                frame_bgr=frame_bgr,
                filename=filename,
                profile=profile,
                runtime_cfg=runtime_cfg,
                rotation_fn=rotation_fn,
                inference=inference,
                model_backend=model_backend,
            )
        except Exception as e:
            # C1 fix: un frame "veleno" non deve fermare la linea. Lo si
            # sposta in quarantena (per poterlo analizzare dopo) e si
            # continua con il frame successivo, invece di far crashare
            # main() e ricadere sullo stesso file al riavvio.
            logger.error(f"Elaborazione fallita per '{filename}': {e!r}. Frame spostato in quarantena.")
            try:
                quarantine_path = os.path.join(quarantine_dir, filename)
                os.replace(frame_path, quarantine_path)
            except OSError as move_err:
                logger.warning(f"Impossibile spostare '{frame_path}' in quarantena ({move_err}).")
            continue

        frame_count += 1
        total_latency_ms += latency_ms
        avg_latency_ms: float = total_latency_ms / frame_count

        logger.info(f"     Average over {frame_count} frames: {avg_latency_ms:.1f} ms")

        try:
            os.remove(frame_path)
        except OSError as e:
            # Prima catturava solo PermissionError: qualsiasi altro OSError
            # (file già rimosso, mount di rete, disco non raggiungibile...)
            # faceva comunque crashare il loop DOPO un'ispezione riuscita.
            # L'ispezione e il salvataggio sono già andati a buon fine qui,
            # quindi si logga e si continua: non c'è nulla da recuperare o
            # da rimettere in coda.
            logger.warning(f"Could not delete '{frame_path}' ({e}).")


if __name__ == "__main__":
    main()