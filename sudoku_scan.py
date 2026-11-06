import cv2
import numpy as np
import os

from corner_detector import CornerDetector
from corner_selection import select_outer_corners


# Input images
image_files = [
    "sudoku_01.jpg",
    "sudoku_02.jpg",
    "sudoku_03.jpg",
    "sudoku_04.jpg",
    "sudoku_05.jpg"
]


# Working size: the longest side is limited to this value, so the pixel
# parameters below are tuned for ~1000 px images (large photos are
# downscaled, small ones are left as they are).
MAX_SIDE = 1000


# Preprocessing parameters (tuned for MAX_SIDE)
blur_ksize = 5
adaptive_block = 31      # must be odd
adaptive_c = 7


# Harris parameters (tuned for MAX_SIDE)
harris_block_size = 3
harris_ksize = 3
harris_k = 0.04
harris_threshold = 0.035
nms_radius = 7


# Sub-pixel refinement (applied only to the 4 selected corners)
# The window is proportional to the working image size.
subpix_win_ratio = 0.006
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)


def save_results(base_dir, results):
    """
    results: {"subfolder/filename.png": image, ...}
    Creates the subfolders and writes every image.
    """
    for rel_path, img in results.items():
        path = os.path.join(base_dir, rel_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, img)


def draw_points(gray, points, color=(0, 0, 255), radius=6):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x, y in points:
        cv2.circle(vis, (int(x), int(y)), radius, color, -1)
    return vis


def normalize_response(resp):
    # Gamma lifts weak responses so the map is readable
    r = np.clip(resp, 0, None)
    if r.max() > 0:
        r = r / r.max()
    return (np.power(r, 0.25) * 255).astype(np.uint8)


def refine_corners(img, pts, win):
    """
    cornerSubPix on each selected point.
    Points too close to the border are left as they are.
    """
    h, w = img.shape[:2]
    refined = []
    for x, y in pts:
        p = np.array([[[x, y]]], np.float32)
        if win + 1 <= x < w - win - 1 and win + 1 <= y < h - win - 1:
            cv2.cornerSubPix(img, p, (win, win), (-1, -1), subpix_criteria)
        refined.append((float(p[0, 0, 0]), float(p[0, 0, 1])))
    return refined


def draw_selected(gray, candidates, selected):
    """All candidates in grey, selected quad with TL/TR/BR/BL labels."""
    vis = draw_points(gray, candidates, color=(160, 160, 160), radius=3)
    if selected is None:
        cv2.putText(vis, "not_found", (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 3)
        return vis
    pts = np.array(selected, np.int32)
    cv2.polylines(vis, [pts.reshape(-1, 1, 2)], True, (0, 200, 0), 2)
    for name, (x, y) in zip(["TL", "TR", "BR", "BL"], selected):
        cv2.circle(vis, (int(x), int(y)), 8, (0, 0, 255), -1)
        cv2.putText(vis, name, (int(x) + 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return vis


def draw_chain(gray, candidates, anchor, chain, color=(0, 200, 0)):
    """
    Dev helper for chain collection: all candidates in grey,
    the chain in colour, the anchor circled in red.
    """
    vis = draw_points(gray, candidates, color=(160, 160, 160), radius=3)
    for x, y in chain:
        cv2.circle(vis, (int(x), int(y)), 6, color, -1)
    cv2.circle(vis, (int(anchor[0]), int(anchor[1])), 10, (0, 0, 255), 2)
    return vis


for image_name in image_files:

    image_path = os.path.join("input", image_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Image could not be loaded: {image_name}")
        continue

    print(f"\nProcessing: {image_name}")

    image_output_name = os.path.splitext(image_name)[0]
    output_dir = os.path.join("output", image_output_name)


    # 0. Downscale to the working size (keep the factor for original coords)
    scale = min(1.0, MAX_SIDE / max(image.shape[:2]))
    if scale < 1.0:
        image = cv2.resize(
            image, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_AREA
        )
    print(f"Working size: {image.shape[1]}x{image.shape[0]} (scale {scale:.3f})")


    # 1. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # 2. Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)


    # 3. Otsu Thresholding (kept for comparison only)
    _, otsu = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )


    # 4. Adaptive Thresholding (normal, for comparison)
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        adaptive_block,
        adaptive_c
    )


    # 5. Grid Line Detection (adaptive inverse: robust to uneven lighting)
    inverted = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        adaptive_block,
        adaptive_c
    )

    height, width = inverted.shape

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(10, width // 40), 1)
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(10, height // 40))
    )

    horizontal_lines = cv2.morphologyEx(
        inverted,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    vertical_lines = cv2.morphologyEx(
        inverted,
        cv2.MORPH_OPEN,
        vertical_kernel
    )

    grid_image = cv2.add(horizontal_lines, vertical_lines)


    # 6. Harris Corner Detection (no dilate on the response: NMS does the job)
    corner_detector = CornerDetector()

    # 6a. On the preprocessed grid mask (main input)
    prepared = cv2.GaussianBlur(
        cv2.dilate(grid_image, np.ones((3, 3), np.uint8)), (5, 5), 0
    )

    harris_grid = cv2.cornerHarris(
        np.float32(prepared) / 255.0,
        blockSize=harris_block_size,
        ksize=harris_ksize,
        k=harris_k
    )

    # 6b. On the raw grayscale image (for comparison)
    harris_gray = cv2.cornerHarris(
        gray.astype(np.float32) / 255.0,
        blockSize=harris_block_size,
        ksize=harris_ksize,
        k=harris_k
    )


    # 7. Detect Harris Candidates (threshold + NMS)
    corner_points = corner_detector.detect_corners(
        harris_grid, harris_threshold, nms_radius
    )
    corner_points_gray = corner_detector.detect_corners(
        harris_gray, harris_threshold, nms_radius
    )

    print(f"Harris candidates (grid mask): {len(corner_points)}")
    print(f"Harris candidates (raw gray):  {len(corner_points_gray)}")


    # Candidates in original-image coordinates (kept for reference; the final
    # output uses the selected and refined corners from step 7b)
    corner_points_original = [(x / scale, y / scale) for x, y in corner_points]


    # 7b. Outer corner selection (own rule) + sub-pixel refinement
    selected = select_outer_corners(corner_points, gray.shape)
    corners_original = None

    if selected is None:
        print("Outer corners: not_found")
    else:
        # Final sanity check only reports; whether to reject is your decision
        ok, info = CornerDetector.check_quad(selected, gray.shape)
        print(f"Quad check: {'ok' if ok else info['reasons']}")

        win = max(3, int(round(subpix_win_ratio * max(gray.shape))))
        selected = refine_corners(blurred, selected, win)

        # Back to original-image coordinates (pixel-centre aware)
        corners_original = [
            ((x + 0.5) / scale - 0.5, (y + 0.5) / scale - 0.5)
            for x, y in selected
        ]
        print(f"Corners (original coords): "
              f"{np.round(corners_original, 1).tolist()}")


    # 8. Draw Harris Candidates
    harris_result = draw_points(gray, corner_points)
    harris_result_gray = draw_points(gray, corner_points_gray)


    # 9. Save Results
    save_results(output_dir, {
        "preprocessing/gray.png": gray,
        "preprocessing/blurred.png": blurred,
        "preprocessing/otsu.png": otsu,
        "preprocessing/adaptive.png": adaptive,
        "preprocessing/adaptive_inv.png": inverted,
        "lines/horizontal_lines.png": horizontal_lines,
        "lines/vertical_lines.png": vertical_lines,
        "lines/combined_grid.png": grid_image,
        "corners/harris_response_grid.png": normalize_response(harris_grid),
        "corners/harris_response_gray.png": normalize_response(harris_gray),
        "corners/harris_corners.png": harris_result,
        "corners/harris_corners_gray_input.png": harris_result_gray,
        "corners/selected_corners.png": draw_selected(gray, corner_points, selected),
    })

    print(f"Results saved for: {image_name}")


print("\nAll images processed.")