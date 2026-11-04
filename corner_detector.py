import cv2
import numpy as np


class CornerDetector:

    # ... (existing methods stay as they are)

    @staticmethod
    def sort_cyclic(pts):
        # Sort points by angle around their centroid (independent of rotation)
        pts = np.asarray(pts, np.float32)
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        return pts[np.argsort(angles)]

    @staticmethod
    def check_quad(pts, img_shape, angle_tol=30.0,
                   min_side_ratio=0.5, min_area_ratio=0.1):
        """
        Check whether 4 points form a reasonable square-like quadrilateral.
        Returns (ok, info). info holds the measured values for debugging.
        """
        pts = CornerDetector.sort_cyclic(pts)
        info = {"reasons": []}

        self.corner_points = [
            (int(x), int(y)) for y, x in points
        ]

        return self.corner_points