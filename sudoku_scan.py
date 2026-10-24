
import cv2

image = cv2.imread("images/sudoku_01.jpg")

if image is None:
    print("Image could not be loaded.")

else:
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
    21,  # blockSize
    5    # C
)

    # Resize images for display
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

    cv2.imwrite("images/gray.png", gray)
    cv2.imwrite("images/blurred.png", blurred)
    cv2.imwrite("images/otsu.png", otsu)
    cv2.imwrite("images/adaptive.png", adaptive)
    
    # Display results
    cv2.imshow("Original", original_display)
    cv2.imshow("Grayscale", gray_display)
    cv2.imshow("Gaussian Blur", blurred_display)
    cv2.imshow("Otsu Threshold", otsu_display)
    cv2.imshow("Adaptive Threshold", adaptive_display)

    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
