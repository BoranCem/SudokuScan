import cv2
import numpy as np
import os
from corner_detector import CornerDetector

image_files = [
    "sudoku_01.jpg",
    "sudoku_02.jpg",
    "sudoku_03.jpg",
    "sudoku_04.jpg",
    "sudoku_05.jpg"
]

# Resize settings (display only)
MAX_WIDTH = 800
MAX_HEIGHT = 600

# Perspective warp output size
WARP_SIZE = 450


def resize_for_display(img, size):
    return cv2.resize(img, size)


for image_name in image_files:

    image_path = "input/" + image_name
    image = cv2.imread(image_path)

    if image is None:
        print(f"Image could not be loaded: {image_name}")
        continue

    print(f"Processing: {image_name}")

    image_output_name = os.path.splitext(image_name)[0]

    preprocessing_dir = f"output/{image_output_name}/preprocessing"
    lines_dir = f"output/{image_output_name}/lines"
    corners_dir = f"output/{image_output_name}/corners"

    for d in (preprocessing_dir, lines_dir, corners_dir, warp_dir):
        os.makedirs(d, exist_ok=True)

    height, width = image.shape[:2]
    scale = min(MAX_WIDTH / width, MAX_HEIGHT / height, 1)
    new_size = (int(width * scale), int(height * scale))

    # 1. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Otsu Thresholding
    _, otsu = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 4. Adaptive Thresholding
    adaptive = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21, 5
    )

    # 5. Invert image
    inverted = cv2.bitwise_not(otsu)

    # 6. Create horizontal and vertical kernels
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(10, width // 40), 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(10, height // 40))
    )

    # 7. Detect horizontal lines
    horizontal_lines = cv2.morphologyEx(
        inverted, cv2.MORPH_OPEN, horizontal_kernel
    )

    # 8. Detect vertical lines
    vertical_lines = cv2.morphologyEx(
        inverted, cv2.MORPH_OPEN, vertical_kernel
    )

    # 9. Combine horizontal and vertical lines
    grid_image = cv2.add(horizontal_lines, vertical_lines)

    # 10. Harris Corner Detection

    # Harris parameters
    harris_block_size = 3
    harris_ksize = 5
    harris_k = 0.04
    harris_threshold = 0.035
    nms_radius = 7

    corner_detector = CornerDetector()

    # Harris runs on the preprocessed grid mask (not on the raw gray image)
    prepared = cv2.GaussianBlur(
        cv2.dilate(grid_image, np.ones((3, 3), np.uint8)), (5, 5), 0
    )

    harris = cv2.cornerHarris(
        np.float32(prepared) / 255.0,
        blockSize=harris_block_size,
        ksize=harris_ksize,
        k=harris_k
    )

    # Harris candidates (threshold + NMS)
    raw_points = corner_detector.detect_corners(
        harris, harris_threshold, nms_radius
    )
    print(f"Detected Harris candidates: {len(raw_points)}")

    # Restrict candidates to the sudoku region
    region = corner_detector.region_mask(grid_image)
    corner_points = corner_detector.filter_by_region(raw_points, region)
    print(f"Candidates inside region: {len(corner_points)}")

    # Sub-pixel refinement
    corner_points = corner_detector.refine_subpix(gray, corner_points)

    # Outer 4 corners (selected among Harris candidates)
    outer = None
    try:
        outer = corner_detector.select_outer_corners(
            corner_points, region, gray.shape
        )

    # 12. Save results
    cv2.imwrite(f"{preprocessing_dir}/gray.png", gray)
    cv2.imwrite(f"{preprocessing_dir}/blurred.png", blurred)
    cv2.imwrite(f"{preprocessing_dir}/otsu.png", otsu)
    cv2.imwrite(f"{preprocessing_dir}/adaptive.png", adaptive)

    # Save line detection results
    cv2.imwrite(f"{lines_dir}/horizontal_lines.png", horizontal_lines)
    cv2.imwrite(f"{lines_dir}/vertical_lines.png", vertical_lines)
    cv2.imwrite(f"{lines_dir}/combined_grid.png", grid_image)

    # Save Harris results
    cv2.imwrite(f"{corners_dir}/harris_corners.png", harris_result)

    # 13. Display results
    cv2.imshow("Original", resize_for_display(image, new_size))
    cv2.imshow("Adaptive Threshold (inv)", resize_for_display(inverted, new_size))
    cv2.imshow("Combined Sudoku Grid", resize_for_display(grid_image, new_size))
    cv2.imshow("Grid Corners", resize_for_display(corners_result, new_size))
    cv2.imshow("Harris Corners", resize_for_display(harris_result, new_size))
    if warped is not None:
        cv2.imshow("Warped", warped)

    print("Press any key to continue.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()