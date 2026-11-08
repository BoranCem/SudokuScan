"""
SudokuScan: photograph of a printed sudoku -> 9x9 digital grid.

    python sudoku_scan.py --input photo.jpg --outdir out/
    python sudoku_scan.py --batch images_dir/ --outdir results/
    python sudoku_scan.py --batch input/ --outdir output/ --debug
"""
import argparse
import os
import time

import cv2
import numpy as np

from corner_detector import CornerDetector, select_outer_corners
import helpers
import rectification
import recognition
import validation


# Working size: the longest side is limited to this value, so the pixel
# parameters below are tuned for ~1000 px images.
MAX_SIDE = 1000

# Preprocessing
blur_ksize = 5
adaptive_block = 31          # must be odd
adaptive_c = 3               # low value keeps thin lines visible in shadows
line_kernel_div = 80         # line kernel length = image side / this value
region_margin_ratio = 0.006  # puzzle region is grown by this * short side

# Harris
harris_block_size = 3
harris_ksize = 3
harris_k = 0.04
harris_threshold = 0.035
nms_radius = 7

# Sub-pixel refinement of the four selected corners
subpix_win_ratio = 0.012
subpix_max_shift_ratio = 0.02


# ---- command line ---------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input", help="one photograph")
parser.add_argument("--batch", help="a folder with photographs")
parser.add_argument("--outdir", default="out", help="output folder")
parser.add_argument("--debug", action="store_true",
                    help="also save the preprocessing / lines / corners / recognition folders")
args = parser.parse_args()

if args.input:
    image_paths = [args.input]
elif args.batch:
    image_paths = [os.path.join(args.batch, f) for f in sorted(os.listdir(args.batch))
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))]
else:
    parser.error("give --input or --batch")

tesseract, message = recognition.load_tesseract()
if message:
    print("WARNING:", message)


for image_path in image_paths:

    start_time = time.time()
    name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"Processing: {name}")

    # Everything that goes into <name>_result.json (a dictionary)
    result = {
        "image": os.path.basename(image_path),
        "corners": None, "A3": None, "A4": None, "H4": None,
        "a3_corner_err_px": None, "a4_rms_px": None,
        "grid": None, "verdict": "not_found", "runtime_s": None,
    }
    images = {}   # files saved next to the result: {"_harris.png": image, ...}
    debug = {}    # extra files, saved only with --debug

    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("image could not be read")

        # 1. Downscale large photos (corners are converted back at the end)
        scale = min(1.0, MAX_SIDE / max(image.shape[:2]))
        if scale < 1.0:
            work = cv2.resize(image, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA)
        else:
            work = image

        # 2. Grayscale and Gaussian blur
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

        # 3. Adaptive threshold, inverted (lines and digits become white)
        inverted = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, adaptive_block, adaptive_c)

        # 4. Grid lines (only used to find the puzzle region)
        height, width = inverted.shape
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(10, width // line_kernel_div), 1))
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(10, height // line_kernel_div)))
        horizontal_lines = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, vertical_kernel)
        grid_image = cv2.add(horizontal_lines, vertical_lines)

        # 5. Harris corner candidates (threshold + non-maximum suppression)
        prepared = cv2.GaussianBlur(
            cv2.dilate(inverted, np.ones((3, 3), np.uint8)), (5, 5), 0)
        harris = cv2.cornerHarris(
            np.float32(prepared) / 255.0,
            blockSize=harris_block_size, ksize=harris_ksize, k=harris_k)
        corner_detector = CornerDetector()
        corner_points = corner_detector.detect_corners(harris, harris_threshold, nms_radius)

        # 6. Keep candidates inside the puzzle region, select the 4 outer
        #    corners among them, refine them to sub-pixel accuracy
        region = helpers.make_region_mask(grid_image, region_margin_ratio)
        candidates = helpers.filter_by_region(corner_points, region)
        outer = select_outer_corners(candidates, gray.shape)

        corners = None
        if outer is not None:
            corners = helpers.refine_corners(
                gray, outer, subpix_win_ratio, subpix_max_shift_ratio)

        images["_harris.png"] = helpers.make_harris_image(
            harris, corner_points, candidates, corners)
        result["n_harris_candidates"] = len(corner_points)
        result["n_region_candidates"] = len(candidates)

        debug["preprocessing/gray.png"] = gray
        debug["preprocessing/blurred.png"] = blurred
        debug["preprocessing/adaptive_inv.png"] = inverted
        debug["lines/horizontal_lines.png"] = horizontal_lines
        debug["lines/vertical_lines.png"] = vertical_lines
        debug["lines/combined_grid.png"] = grid_image
        debug["lines/region_mask.png"] = region
        debug["corners/harris_corners.png"] = helpers.draw_points(gray, corner_points)
        selected_view = helpers.draw_points(gray, candidates, (0, 0, 255), 4)
        if corners is not None:
            for x, y in corners:
                cv2.circle(selected_view, (int(round(x)), int(round(y))), 14, (255, 0, 0), 3)
        debug["corners/selected.png"] = selected_view

        if corners is None:
            raise ValueError("outer corners not found")

        # 7. Rectification: A3, A4 and H4 (450x450)
        models = rectification.estimate_models(corners)
        warped = rectification.warp_all(work, models)
        images["_rect_A3.png"] = warped["A3"]
        images["_rect_A4.png"] = warped["A4"]
        images["_rect_H4.png"] = warped["H4"]

        # results in original-image coordinates
        result["corners"] = (corners / scale).tolist()
        result["A3"] = rectification.to_original_coordinates(models["A3"], scale).tolist()
        result["A4"] = rectification.to_original_coordinates(models["A4"], scale).tolist()
        result["H4"] = rectification.to_original_coordinates(models["H4"], scale).tolist()
        result["a3_corner_err_px"] = round(models["a3_corner_err_px"], 4)
        result["a4_rms_px"] = round(models["a4_rms_px"], 4)
        result["h4_max_corner_err_px"] = float(models["h4_max_corner_err_px"])

        # 8. Cells, empty/filled decision and digit recognition (on H4)
        rect_gray = cv2.cvtColor(warped["H4"], cv2.COLOR_BGR2GRAY)
        recognised = recognition.recognise_grid(rect_gray, pytesseract=tesseract)
        grid = recognised["grid"]
        result["grid"] = grid
        if tesseract is None:
            result["warnings"] = ["OCR engine unavailable: digits were not read"]
        debug["recognition/ink.png"] = recognised["ink"]
        debug["recognition/ocr_inputs.png"] = recognition.ocr_contact_sheet(
            recognised["ocr_inputs"])

        # 9. Validity check (row, column, 3x3 box)
        is_valid, conflicts = validation.check_validity(grid)
        result["verdict"] = "valid" if is_valid else "invalid"
        result["conflicts"] = conflicts
        images["_debug.png"] = helpers.make_debug_image(warped["H4"], grid, conflicts)

    except Exception as error:      # one bad image must never stop the batch
        result["error"] = str(error)

    # 10. Save the output files
    result["runtime_s"] = round(time.time() - start_time, 3)
    helpers.save_outputs(args.outdir, name, result, images)
    if args.debug:
        helpers.save_images(os.path.join(args.outdir, name), debug)

    print(f"  -> {result['verdict']} ({result['runtime_s']} s)")