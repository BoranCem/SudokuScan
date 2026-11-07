import cv2
import numpy as np
import os

from corner_detector import CornerDetector, select_outer_corners


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


# Line mask (used ONLY to find the puzzle region, not as Harris input).
# Kernel length = image side / line_kernel_div. Longer kernels remove digits
# but break rotated / slanted lines, so this goes into the parameter study.
line_kernel_div = 80
region_margin_ratio = 0.01   # region is grown by this fraction of the short side


# Harris parameters (tuned for MAX_SIDE)
harris_block_size = 3
harris_ksize = 3
harris_k = 0.04
harris_threshold = 0.035
nms_radius = 7


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


def make_region_mask(grid_mask, margin_ratio=region_margin_ratio):
    """
    Puzzle region from the line mask. Gaps are closed with a small kernel and
    only the LARGEST connected component (the grid) is kept, so printed labels
    such as "9x9" / "zor" and neighbouring tables, which are separate
    components, stay outside. The convex hull of that component is filled and
    grown a little. This is only a region; the corner coordinates still come
    from Harris candidates.
    """
    h, w = grid_mask.shape[:2]
    k = max(3, int(round(min(h, w) * 0.01)) | 1)
    closed = cv2.morphologyEx(
        grid_mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )

    region = np.zeros((h, w), np.uint8)
    if n <= 1:
        return region

    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(labels == biggest)
    hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32))
    cv2.fillConvexPoly(region, hull, 255)

    m = max(3, int(round(min(h, w) * margin_ratio)))
    return cv2.dilate(region, np.ones((2 * m + 1, 2 * m + 1), np.uint8))


def filter_by_region(points, region):
    h, w = region.shape[:2]
    return [
        (x, y) for x, y in points
        if 0 <= int(x) < w and 0 <= int(y) < h and region[int(y), int(x)] > 0
    ]


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
        (max(10, width // line_kernel_div), 1)
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(10, height // line_kernel_div))
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

    # 6a. On the adaptive-inverse mask WITHOUT line opening (main input):
    # opening with horizontal/vertical kernels breaks rotated and slanted
    # lines, which removes the candidates along the puzzle edges.
    prepared = cv2.GaussianBlur(
        cv2.dilate(inverted, np.ones((3, 3), np.uint8)), (5, 5), 0
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


    # 7b. Restrict candidates to the puzzle region (line mask -> region only)
    region = make_region_mask(grid_image)
    candidates = filter_by_region(corner_points, region)
    print(f"Candidates inside region:      {len(candidates)}")


    # 7c. Select the 4 outer corners among the Harris candidates
    outer = select_outer_corners(candidates, gray.shape)

    if outer is None:
        print("Outer corners: NOT FOUND")
        outer_original = None
    else:
        # Back to original-image coordinates (needed for the final output)
        outer_original = [(x / scale, y / scale) for x, y in outer]
        print("Outer corners (working size): "
              + ", ".join(f"({x:.1f}, {y:.1f})" for x, y in outer))
        print("Outer corners (original):     "
              + ", ".join(f"({x:.1f}, {y:.1f})" for x, y in outer_original))


    # 8. Draw Harris Candidates
    harris_result = draw_points(gray, corner_points)
    harris_result_gray = draw_points(gray, corner_points_gray)

    # Region contour (green), surviving candidates (red), selected corners (blue)
    selected_result = draw_points(gray, candidates, (0, 0, 255), 4)
    region_contours, _ = cv2.findContours(
        region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(selected_result, region_contours, -1, (0, 255, 0), 2)
    if outer is not None:
        for x, y in outer:
            cv2.circle(selected_result, (int(x), int(y)), 14, (255, 0, 0), 3)


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
        "corners/harris_corners.png": harris_result,
        "corners/harris_corners_gray_input.png": harris_result_gray,
        "corners/selected.png": selected_result,
        "lines/region_mask.png": region,
    })

    print(f"Results saved for: {image_name}")


print("\nAll images processed.")