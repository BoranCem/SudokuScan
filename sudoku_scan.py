import cv2

image = cv2.imread("images/sudoku_01.jpg")

if image is None:
    print("Image could not be loaded.")
    print("Check the image path: images/sudoku_01.jpg")
else:
    height, width = image.shape[:2]
    print(f"Image loaded: {width} x {height}")

    max_width = 800
    max_height = 600

    scale = min(max_width / width, max_height / height, 1)

    new_width = int(width * scale)
    new_height = int(height * scale)

    display_image = cv2.resize(
        image, (new_width, new_height)
    )

    cv2.imshow("Original Sudoku", display_image)
    print("Press any key while the image window is selected.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()