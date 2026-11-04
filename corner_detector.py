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

        # 1) Side lengths
        sides = np.array([
            np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)
        ])
        info["sides"] = sides.round(1).tolist()
        if sides.min() == 0:
            info["reasons"].append("duplicate points")
            return False, info
        info["side_ratio"] = round(float(sides.min() / sides.max()), 3)

        # 2) Interior angles at each vertex
        angles = []
        for i in range(4):
            a = pts[i - 1] - pts[i]
            b = pts[(i + 1) % 4] - pts[i]
            cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            angles.append(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))))
        info["angles"] = [round(a, 1) for a in angles]

        # 3) Convexity: cross products of consecutive edges share one sign
        crosses = []
        for i in range(4):
            e1 = pts[(i + 1) % 4] - pts[i]
            e2 = pts[(i + 2) % 4] - pts[(i + 1) % 4]
            crosses.append(e1[0] * e2[1] - e1[1] * e2[0])
        convex = all(c > 0 for c in crosses) or all(c < 0 for c in crosses)
        info["convex"] = convex
