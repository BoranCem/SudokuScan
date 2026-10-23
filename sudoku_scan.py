
import cv2

image = cv2.imread("images/sudoku_01.jpg")

if image is None:
    print("Image could not be loaded.")

else:
    # 1. Resize only for display
    max_width = 800
    max_height = 600

    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1)

    new_width = int(width * scale)
    new_height = int(height * scale)

    display_original = cv2.resize(
        image, (new_width, new_height)
    )

    # 2. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. Reduce noise with Gaussian Blur
    blurred = cv2.GaussianBlur(
        gray, (5, 5), 0
    )

    # Resize processed images for display
    display_gray = cv2.resize(
        gray, (new_width, new_height)
    )

    display_blurred = cv2.resize(
        blurred, (new_width, new_height)
    )

    # 4. Show results
    cv2.imshow("1 - Original", display_original)
    cv2.imshow("2 - Grayscale", display_gray)
    cv2.imshow("3 - Gaussian Blur", display_blurred)

    print("Press any key to close the windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
