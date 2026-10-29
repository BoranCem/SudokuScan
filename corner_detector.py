import cv2
import numpy as np


class CornerDetector:

    def __init__(self):
        self.corner_points = []
        self.selected_corners = []

    def detect_corners(self, harris):
        threshold = 0.035 * harris.max()

        points = np.argwhere(harris > threshold)

        self.corner_points = [
            (int(x), int(y)) for y, x in points
        ]

        return self.corner_points