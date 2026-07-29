"""
alignment.py (OPTIMIZED)
=====================
Modulo per l'allineamento verticale dei tappi.
Ottimizzato per elaborazione in memoria in tempo reale.
"""

import math
import cv2
import numpy as np

import time


def _safe_crop(img: np.ndarray, cx: int, cy: int, half: int) -> tuple[np.ndarray, int, int]:
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


def _estimate_cap_center(gray: np.ndarray) -> tuple[float, float] | None:
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    (cx, cy), _ = cv2.minEnclosingCircle(biggest)
    return (float(cx), float(cy))


def _best_pair_from_candidates(
    candidates: np.ndarray,
    cap_center: tuple[float, float] | None,
    pair_dist_range: tuple[float, float],
    center_dist_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray] | None:
    if candidates is None or len(candidates) < 2:
        return None

    if cap_center is not None:
        ccx, ccy = cap_center
        scored = [(c, float(np.hypot(c[0] - ccx, c[1] - ccy))) for c in candidates]
    else:
        scored = [(c, None) for c in candidates]

    best_pair = None
    best_score = float("inf")
    for i in range(len(scored)):
        for j in range(i + 1, len(scored)):
            c1, d1 = scored[i]
            c2, d2 = scored[j]
            x1, y1, r1 = c1
            x2, y2, r2 = c2

            pair_dist = float(np.hypot(x1 - x2, y1 - y2))
            if not (pair_dist_range[0] <= pair_dist <= pair_dist_range[1]):
                continue

            if cap_center is not None:
                if not (center_dist_range[0] <= d1 <= center_dist_range[1]):
                    continue
                if not (center_dist_range[0] <= d2 <= center_dist_range[1]):
                    continue
                score = abs(r1 - r2) + abs(d1 - d2)
            else:
                score = abs(r1 - r2)

            if score < best_score:
                best_score = score
                best_pair = (c1, c2)

    return best_pair


def _find_marker_pair(
    gray: np.ndarray,
    min_radius: int,
    max_radius: int,
    min_dist: int,
    param1: float,
    param2: float,
    pair_dist_range: tuple[float, float] = (700.0, 860.0),
    center_dist_range: tuple[float, float] = (300.0, 470.0),
) -> tuple[np.ndarray, np.ndarray] | None:
    blurred = cv2.medianBlur(gray, 5)
    cap_center = _estimate_cap_center(gray)

    attempts = [
        dict(param2=param2, minRadius=min_radius, maxRadius=max_radius),
        dict(param2=param2 - 5, minRadius=min_radius - 5, maxRadius=max_radius + 5),
        dict(param2=param2 - 10, minRadius=min_radius - 10, maxRadius=max_radius + 10),
        dict(param2=max(param2 - 15, 10), minRadius=min_radius - 15, maxRadius=max_radius + 15),
    ]

    for att in attempts:
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist,
            param1=param1, param2=att["param2"],
            minRadius=max(1, att["minRadius"]), maxRadius=att["maxRadius"],
        )
        if circles is None:
            continue
        pair = _best_pair_from_candidates(circles[0], cap_center, pair_dist_range, center_dist_range)
        if pair is not None:
            return pair

    return None


def find_cap_center_via_markers(
    frame_bgr: np.ndarray,
    min_radius: int = 76,
    max_radius: int = 100,
    min_dist: int = 600,
    param1: float = 50,
    param2: float = 30,
    pair_dist_range: tuple[float, float] = (700.0, 860.0),
    center_dist_range: tuple[float, float] = (300.0, 470.0),
) -> tuple[float, float] | None:
    """
    Stima il centro del tappo come punto medio tra i due marker fisici
    rilevati via Hough (stessa logica robusta di allineamento usata in
    align_cap_image/align_cap_image_small), invece della soglia fissa di
    src.slot_detection (pensata per il tappo rosso opaco e non adatta a
    tappi trasparenti).

    Pensata per essere riusata da augment_dataset.py: qui interessa SOLO
    il centro (per ruotare attorno al punto giusto durante la generazione
    del dataset sintetico), non l'angolo/allineamento finale.

    Ritorna None se non è stato possibile rilevare una coppia di marker
    plausibile (stesso criterio fail-soft di align_cap_image: chi chiama
    decide cosa fare, es. scartare l'immagine sorgente).
    """
    if frame_bgr is None:
        return None

    gray: np.ndarray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr

    pair = _find_marker_pair(
        gray,
        min_radius=min_radius, max_radius=max_radius,
        min_dist=min_dist, param1=param1, param2=param2,
        pair_dist_range=pair_dist_range, center_dist_range=center_dist_range,
    )
    if pair is None:
        return None

    c1, c2 = pair
    cx = (float(c1[0]) + float(c2[0])) / 2.0
    cy = (float(c1[1]) + float(c2[1])) / 2.0
    return (cx, cy)


def align_cap_image(img: np.ndarray, template: np.ndarray) -> np.ndarray:
    if img is None or template is None:
        return img

    t0 = time.perf_counter()

    gray: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    pair = _find_marker_pair(
        gray,
        min_radius=76, max_radius=100,
        min_dist=600, param1=50, param2=30,
    )

    if pair is None:
        print("[WARNING] Impossibile rilevare 2 marker plausibili. Rotazione saltata.")
        return img

    c1, c2 = pair

    x1, y1 = int(c1[0]), int(c1[1])
    x2, y2 = int(c2[0]), int(c2[1])

    delta_y: int = y2 - y1
    delta_x: int = x2 - x1
    current_angle: float = math.degrees(math.atan2(delta_y, delta_x))
    base_rotation: float = current_angle - 90

    height, width = img.shape[:2]
    cap_center_x: int = (x1 + x2) // 2
    cap_center_y: int = (y1 + y2) // 2
    centre: tuple[int, int] = (cap_center_x, cap_center_y)

    warp_flags: int = cv2.INTER_CUBIC
    border_mode: int = cv2.BORDER_REPLICATE

    radius_estimate = max(int(c1[2]), int(c2[2]), 100)
    template_h, template_w = template.shape[:2]
    margin = max(template_h, template_w) // 2 + 20

    half_crop = int((radius_estimate + margin) * 1.45)
    half_crop = max(half_crop, max(template_h, template_w))

    gray_crop, cx_local, cy_local = _safe_crop(gray, cap_center_x, cap_center_y, half_crop)
    centre_local = (cx_local, cy_local)
    crop_side = gray_crop.shape[0]

    M_upright_crop: np.ndarray = cv2.getRotationMatrix2D(centre_local, base_rotation, 1.0)
    M_flipped_crop: np.ndarray = cv2.getRotationMatrix2D(centre_local, base_rotation + 180, 1.0)

    crop_upright: np.ndarray = cv2.warpAffine(
        gray_crop, M_upright_crop, (crop_side, crop_side),
        flags=warp_flags, borderMode=border_mode,
    )
    crop_flipped: np.ndarray = cv2.warpAffine(
        gray_crop, M_flipped_crop, (crop_side, crop_side),
        flags=warp_flags, borderMode=border_mode,
    )

    score_upright: float = cv2.minMaxLoc(
        cv2.matchTemplate(crop_upright, template, cv2.TM_CCOEFF_NORMED)
    )[1]
    score_flipped: float = cv2.minMaxLoc(
        cv2.matchTemplate(crop_flipped, template, cv2.TM_CCOEFF_NORMED)
    )[1]

    final_angle: float = base_rotation
    if score_flipped > score_upright:
        final_angle += 180

    M_final: np.ndarray = cv2.getRotationMatrix2D(centre, final_angle, 1.0)
    result: np.ndarray = cv2.warpAffine(
        img, M_final, (width, height), flags=warp_flags, borderMode=border_mode
    )

    return result