import cv2
import numpy as np


class CornerDetector:

    def __init__(self):
        self.corner_points = []
        self.selected_corners = []

    # ------------------------------------------------------------------
    # Harris response -> discrete candidates (relative threshold + NMS)
    # ------------------------------------------------------------------
    def detect_corners(self, harris, rel_threshold=0.035, nms_radius=7):
        hmax = float(harris.max())
        if hmax <= 0:
            self.corner_points = []
            return self.corner_points

        # A pixel survives only if it is the maximum of its neighbourhood
        k = 2 * int(nms_radius) + 1
        local_max = cv2.dilate(harris, np.ones((k, k), np.uint8))
        mask = ((harris >= local_max) & (harris > rel_threshold * hmax)).astype(np.uint8)

        # Equal-valued neighbouring pixels (plateaus) become one point
        _, _, _, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        self.corner_points = [
            (int(round(x)), int(round(y))) for x, y in centroids[1:]
        ]
        return self.corner_points

    # ------------------------------------------------------------------
    # Helpers for the corner selection rule
    # ------------------------------------------------------------------
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

        # 4) Area (shoelace formula) relative to the image area
        x, y = pts[:, 0], pts[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        info["area_ratio"] = round(float(area / (img_shape[0] * img_shape[1])), 3)

        # Decision
        if not convex:
            info["reasons"].append("not convex")
        if any(abs(a - 90) > angle_tol for a in angles):
            info["reasons"].append(f"angle off by more than {angle_tol} deg")
        if info["side_ratio"] < min_side_ratio:
            info["reasons"].append("sides too different")
        if info["area_ratio"] < min_area_ratio:
            info["reasons"].append("area too small")

        return len(info["reasons"]) == 0, info

    @staticmethod
    def estimate_cell_size(pts):
        """
        Estimates the true cell size by looking at structured grid distances 
        rather than immediate noisy plateaus.
        """
        if len(pts) < 10:
            return 20.0 # Fallback default
        
        # Calculate pairwise distances
        pts_np = np.array(pts, dtype=np.float32)
        dists_matrix = np.linalg.norm(pts_np[:, None, :] - pts_np[None, :, :], axis=-1)
        
        # For each point, find small non-zero distances which typically represent cell steps
        min_dists = []
        for row in dists_matrix:
            valid = row[row > 2.0] # Exclude self and ultra-close duplicates
            if len(valid) > 0:
                min_dists.append(np.min(valid))
                
        # The median of these minimums gives a much cleaner estimate of cell step
        return float(np.median(min_dists))

    @staticmethod
    def collect_chain(start_pt, direction, pts_set, cell_size, max_gap_ratio=1.5):
        """
        Walks along a specific direction vector from a start point, 
        gathering connected points into a chain.
        """
        chain = [start_pt]
        current = np.array(start_pt, dtype=np.float32)
        dir_norm = direction / (np.linalg.norm(direction) + 1e-5)
        
        step_len = cell_size
        max_step = step_len * max_gap_ratio
        
        while True:
            # Predict next expected position
            expected = current + dir_norm * step_len
            
            # Find candidates near the expected position within tolerance
            best_next = None
            min_dist = max_step
            
            for p in pts_set:
                if tuple(p) in [tuple(c) for c in chain]:
                    continue
                d = np.linalg.norm(p - expected)
                if d < min_dist:
                    min_dist = d
                    best_next = p
                    
            if best_next is not None:
                chain.append(tuple(best_next))
                current = np.array(best_next, dtype=np.float32)
            else:
                break
                
        return chain


def select_outer_corners(corner_points, img_shape):
    """
    Automatically selects the 4 outer corners of the Sudoku puzzle from Harris candidates.
    Returns: list of 4 tuples [(x, y), ...] in (TL, TR, BR, BL) order, or None.
    """
    if len(corner_points) < 4:
        return None

    pts = np.array(corner_points, dtype=np.float32)

    # 1. Estimate cell size / neighborhood scale (median of nearest-neighbor distances)
    # Calculate the distance of each point to its nearest neighbor
    distances = []
    for i, p in enumerate(pts):
        dists = np.linalg.norm(pts - p, axis=1)
        dists = dists[dists > 0] # Exclude self
        if len(dists) > 0:
            distances.append(np.min(dists))
    
    if not distances:
        return None
    
    cell_size = np.median(distances)
    
    # 2. Rough approach: Select the outermost corner candidates covering the Sudoku boundaries.
    # Find the center point
    center = pts.mean(axis=0)
    
    # Select points representing the 4 quadrants (TL, TR, BR, BL) furthest from the center
    tl_cands = pts[(pts[:, 0] < center[0]) & (pts[:, 1] < center[1])]
    tr_cands = pts[(pts[:, 0] >= center[0]) & (pts[:, 1] < center[1])]
    br_cands = pts[(pts[:, 0] >= center[0]) & (pts[:, 1] >= center[1])]
    bl_cands = pts[(pts[:, 0] < center[0]) & (pts[:, 1] >= center[1])]
    
    # Choose the point furthest from the center in each region
    selected = []
    for region_pts in [tl_cands, tr_cands, br_cands, bl_cands]:
        if len(region_pts) == 0:
            return None
        # The point furthest from the center is the outer corner candidate
        dists_to_center = np.linalg.norm(region_pts - center, axis=1)
        best_pt = region_pts[np.argmax(dists_to_center)]
        selected.append(best_pt)
        
    return [tuple(pt) for pt in selected]