"""
Stage 5: cell extraction, empty-cell decision and digit recognition.

Pipeline on the 450x450 homography-rectified grayscale image:
  1. illumination normalisation (divide by a closing-based background)
  2. global ink mask (clamped Otsu on the normalised image)
  3. for each of the 81 cells: crop an inner margin, drop line remnants that
     hug the crop border, decide EMPTY vs FILLED with an explicit criterion
     (largest connected component), and only then call OCR for filled cells
  4. prepare the digit (anti-aliased grey crop of the digit only, centred,
     scaled, white padding) and run Tesseract restricted to the digits 1-9
"""
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

os.environ.setdefault("OMP_THREAD_LIMIT", "1")   # one thread per tesseract call

CELL = 50
GRID = 9

# ---- parameters (for a 50 px cell) -------------------------------------
CELL_MARGIN_RATIO = 0.12         # inner margin on every side: 6 px
EMPTY_AREA_RATIO = 0.02          # largest component >= 2 % of the cropped cell
EMPTY_MIN_HEIGHT_RATIO = 0.30    # ... and >= 30 % of the crop height
NOISE_AREA_RATIO = 0.004         # components smaller than this are noise
OCR_CANVAS = 96                  # square canvas given to the OCR engine
OCR_DIGIT_HEIGHT = 56            # digit height on that canvas
OCR_DIGIT_MAX_WIDTH = 70
OCR_BASE = ("--oem 3 -c tessedit_char_whitelist=123456789 "
            "-c load_system_dawg=0 -c load_freq_dawg=0")
# last resort for a filled cell that the whitelisted runs could not read:
# a few letters that Tesseract returns for badly shaped digits
OCR_LOOKALIKES = {"S": 5, "s": 5, "B": 8, "I": 1, "l": 1, "|": 1,
                  "Z": 2, "z": 2, "G": 6, "g": 9, "q": 9}


# ---- tesseract ----------------------------------------------------------
def load_tesseract(cmd=None):
    """Returns (pytesseract_module, None) or (None, error_message)."""
    try:
        import pytesseract
    except ImportError:
        return None, "pytesseract is not installed (pip install pytesseract)"

    candidates = [
        cmd,
        os.environ.get("TESSERACT_CMD"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            pytesseract.pytesseract.tesseract_cmd = c
            break
    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        return None, ("Tesseract engine not found (%s). Install it or pass "
                      "--tesseract-cmd / set TESSERACT_CMD." % type(exc).__name__)
    return pytesseract, None


def _first_digit(text):
    for ch in text:
        if ch in "123456789":
            return int(ch)
    return 0


def ocr_digit(img, pytesseract):
    """
    Single character (psm 10); if nothing is returned retry as one word
    (psm 8) and as one block (psm 6), all restricted to 1-9. A filled cell
    that is still unread gets one unrestricted run, whose result is mapped
    through OCR_LOOKALIKES. Returns 0 when everything fails.
    """
    for psm in (10, 8, 6):
        d = _first_digit(pytesseract.image_to_string(img, config=f"{OCR_BASE} --psm {psm}"))
        if d:
            return d
    text = pytesseract.image_to_string(img, config="--oem 3 --psm 10")
    for ch in text.strip():
        if ch in OCR_LOOKALIKES:
            return OCR_LOOKALIKES[ch]
        if ch in "123456789":
            return int(ch)
    return 0


# ---- image preparation --------------------------------------------------
def normalize_illumination(gray):
    """Divide by a background estimate (closing removes dark ink and lines)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (0, 0), 8)
    return cv2.divide(gray, np.maximum(background, 1), scale=255)


def binarize_ink(norm):
    """Ink mask (255 = dark). Otsu threshold clamped to a sane range."""
    t, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = float(np.clip(t, 90, 190))
    return (norm < t).astype(np.uint8) * 255


def analyse_cell(ink_crop):
    """
    Removes line remnants and noise from a cropped cell and decides whether
    it is filled.

    Returns (filled, clean_mask, area_ratio, height_ratio) where the ratios
    describe the largest remaining component.
    """
    H, W = ink_crop.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink_crop, connectivity=8)

    clean = np.zeros_like(ink_crop)
    best_area, best_h = 0, 0
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])

        # a thin component lying along one crop edge is a grid-line remnant
        hugs_horizontal = w >= 0.8 * W and h <= 0.35 * H and (y <= 1 or y + h >= H - 1)
        hugs_vertical = h >= 0.8 * H and w <= 0.35 * W and (x <= 1 or x + w >= W - 1)
        if hugs_horizontal or hugs_vertical:
            continue
        if area < NOISE_AREA_RATIO * H * W:
            continue

        clean[labels == i] = 255
        if area > best_area:
            best_area, best_h = area, h

    filled = (best_area >= EMPTY_AREA_RATIO * H * W
              and best_h >= EMPTY_MIN_HEIGHT_RATIO * H)
    return filled, clean, best_area / float(H * W), best_h / float(H)


def prepare_for_ocr(norm_crop, clean_mask):
    """
    Grey-level image of the digit only: pixels outside the (slightly grown)
    digit mask are set to white, which removes line remnants and noise but
    keeps the anti-aliased stroke edges (a binary mask makes strokes too bold
    for the OCR engine). The crop is stretched to full contrast, scaled to a
    fixed height and centred on a white square canvas.
    """
    grown = cv2.dilate(clean_mask, np.ones((5, 5), np.uint8))
    gray = np.where(grown > 0, norm_crop, 255).astype(np.uint8)
    ys, xs = np.nonzero(grown)
    gray = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    lo = int(gray.min())
    gray = np.clip((gray.astype(np.float32) - lo) * 255.0 / max(1, 255 - lo),
                   0, 255).astype(np.uint8)

    h, w = gray.shape
    s = min(OCR_DIGIT_HEIGHT / float(h), OCR_DIGIT_MAX_WIDTH / float(w))
    new_w, new_h = max(1, int(round(w * s))), max(1, int(round(h * s)))
    gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((OCR_CANVAS, OCR_CANVAS), 255, np.uint8)
    oy, ox = (OCR_CANVAS - new_h) // 2, (OCR_CANVAS - new_w) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = gray
    return canvas


# ---- whole grid ---------------------------------------------------------
def recognise_grid(rect_gray, pytesseract=None, prepare=True, workers=4):
    """
    rect_gray: 450x450 grayscale rectified image.
    prepare=False feeds the raw 50x50 cell to OCR (baseline for the
    with/without-preparation comparison). The empty/filled decision is the
    same in both modes and never depends on OCR.
    """
    norm = normalize_illumination(rect_gray)
    ink = binarize_ink(norm)
    m = int(round(CELL * CELL_MARGIN_RATIO))

    grid = [[0] * GRID for _ in range(GRID)]
    filled = [[False] * GRID for _ in range(GRID)]
    stats, jobs = {}, []

    for r in range(GRID):
        for c in range(GRID):
            y0, x0 = r * CELL, c * CELL
            crop = ink[y0 + m:y0 + CELL - m, x0 + m:x0 + CELL - m]
            norm_crop = norm[y0 + m:y0 + CELL - m, x0 + m:x0 + CELL - m]
            is_filled, clean, area_ratio, height_ratio = analyse_cell(crop)
            filled[r][c] = bool(is_filled)
            stats[(r, c)] = (round(area_ratio, 4), round(height_ratio, 3))
            if is_filled:
                img = (prepare_for_ocr(norm_crop, clean) if prepare
                       else rect_gray[y0:y0 + CELL, x0:x0 + CELL])
                jobs.append(((r, c), img))

    if pytesseract is not None and jobs:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            digits = list(pool.map(lambda job: ocr_digit(job[1], pytesseract), jobs))
        for ((r, c), _), d in zip(jobs, digits):
            grid[r][c] = d

    return {
        "grid": grid,
        "filled": filled,
        "ink": ink,
        "cell_stats": stats,                       # (area_ratio, height_ratio)
        "ocr_inputs": {rc: img for rc, img in jobs},
    }


def ocr_contact_sheet(ocr_inputs, tile=64):
    """9x9 sheet of the images that were given to OCR (debug)."""
    sheet = np.full((GRID * tile, GRID * tile), 255, np.uint8)
    for (r, c), img in ocr_inputs.items():
        sheet[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = cv2.resize(
            img, (tile, tile), interpolation=cv2.INTER_AREA)
    for i in range(GRID + 1):
        cv2.line(sheet, (i * tile, 0), (i * tile, GRID * tile), 180, 1)
        cv2.line(sheet, (0, i * tile), (GRID * tile, i * tile), 180, 1)
    return sheet
