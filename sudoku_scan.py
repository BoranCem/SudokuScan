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

    os.makedirs(preprocessing_dir, exist_ok=True)
    os.makedirs(lines_dir, exist_ok=True)
    os.makedirs(corners_dir, exist_ok=True)


    # Resize settings (display only)
    max_width = 800
    max_height = 600

    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1)

    new_width = int(width * scale)
    new_height = int(height * scale)

    # 1. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Otsu Thresholding
    _, otsu = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 4. Adaptive Thresholding
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        5
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
    grid_image = cv2.add(
    horizontal_lines,
    vertical_lines
    )

    # 10. Harris Corner Detection

    # Harris parameters
    harris_block_size = 3
    harris_ksize = 5
    harris_k = 0.04
    harris_threshold = 0.035

    gray_float = gray.astype(np.float32)

    corner_detector = CornerDetector()

    harris = cv2.cornerHarris(
        gray_float,
        blockSize=harris_block_size,
        ksize=harris_ksize,
        k=harris_k
    )

    # Dilate Harris response for better visualization
    harris = cv2.dilate(harris, None)

    corner_points = corner_detector.detect_corners(harris)

    print(f"Detected Harris candidates: {len(corner_points)}")

    # Create a copy of grayscale image
    harris_result = cv2.cvtColor(
    gray, cv2.COLOR_GRAY2BGR
    )

    # Draw larger red circles
    for x, y in corner_points:
        cv2.circle(
            harris_result,
            (x, y),
            6,
            (0, 0, 255),
            -1
        )

    # 11. Resize images for display
    original_display = cv2.resize(
        image, (new_width, new_height)
    )

    gray_display = cv2.resize(
        gray, (new_width, new_height)
    )

    blurred_display = cv2.resize(
        blurred, (new_width, new_height)
    )

    otsu_display = cv2.resize(
        otsu, (new_width, new_height)
    )

    adaptive_display = cv2.resize(
        adaptive, (new_width, new_height)
    )

    horizontal_display = cv2.resize(
        horizontal_lines, (new_width, new_height)
    )

    vertical_display = cv2.resize(
        vertical_lines, (new_width, new_height)
    )

    grid_display = cv2.resize(
    grid_image, (new_width, new_height)
    )

    harris_display = cv2.resize(
        harris_result, (new_width, new_height)
    )

    # 12. Save processed images

    # Save preprocessing results
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
    cv2.imshow("Original", original_display)
    cv2.imshow("Grayscale", gray_display)
    cv2.imshow("Gaussian Blur", blurred_display)
    cv2.imshow("Otsu Threshold", otsu_display)
    cv2.imshow("Adaptive Threshold", adaptive_display)
    cv2.imshow("Horizontal Lines", horizontal_display)
    cv2.imshow("Vertical Lines", vertical_display)
    cv2.imshow("Combined Sudoku Grid", grid_display)
    cv2.imshow("Harris Corners", harris_display)

    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
