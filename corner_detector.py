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
    # Helpers for the corner selection rule & Chain Walking
    # ------------------------------------------------------------------
    @staticmethod
    def estimate_cell_size(pts, img_shape=None):
        """
        Scale-aware diagnostic estimate of the typical grid spacing.

        Outer-corner selection does not depend on this estimate; it derives
        the step size separately from each detected Sudoku edge.
        """
        if len(pts) < 2:
            return None

        pts_np = np.asarray(pts, dtype=np.float32)
        distances = np.linalg.norm(
            pts_np[:, None, :] - pts_np[None, :, :], axis=2
        )
        np.fill_diagonal(distances, np.inf)

        if img_shape is not None:
            diagonal = float(np.hypot(img_shape[0], img_shape[1]))
        else:
            span = np.ptp(pts_np, axis=0)
            diagonal = float(np.hypot(span[0], span[1]))

        # Relative limits keep the estimate usable at different resolutions.
        min_distance = max(2.0, 0.012 * diagonal)
        max_distance = 0.20 * diagonal

        nearest = np.min(distances, axis=1)
        valid = nearest[(nearest >= min_distance) & (nearest <= max_distance)]

        if len(valid) == 0:
            return None

        return float(np.median(valid))

    @staticmethod
    def collect_chain(
        start_pt,
        direction,
        pts_set,
        cell_size,
        max_gap_ratio=2.1,
        lateral_tol_ratio=0.22,
        max_steps=12,
        target_distance=None,
    ):
        """
        Follow candidate points in one direction without jumping to
        neighbouring rows/columns.

        Forward distance and perpendicular (lateral) distance are checked
        independently. A missing grid intersection may be skipped, but a
        point on a neighbouring row/column is rejected.
        """
        start = np.asarray(start_pt, dtype=np.float32)
        direction = np.asarray(direction, dtype=np.float32)
        direction_length = float(np.linalg.norm(direction))

        if direction_length < 1e-6 or cell_size <= 0:
            return [tuple(start)]

        unit = direction / direction_length
        points = np.asarray(pts_set, dtype=np.float32).reshape(-1, 2)

        chain = [tuple(start)]
        used = {tuple(np.round(start, 3))}
        lateral_limit = lateral_tol_ratio * cell_size
        min_forward = 0.45 * cell_size
        max_forward = max_gap_ratio * cell_size

        for _ in range(int(max_steps)):
            current = np.asarray(chain[-1], dtype=np.float32)
            best_point = None
            best_score = float("inf")

            for point in points:
                key = tuple(np.round(point, 3))
                if key in used:
                    continue

                delta = point - current
                forward = float(np.dot(delta, unit))

                # Only consider points genuinely ahead of the current point.
                if forward < min_forward or forward > max_forward:
                    continue

                # Perpendicular distance from the intended chain direction.
                lateral = abs(
                    float(unit[0] * delta[1] - unit[1] * delta[0])
                )
                if lateral > lateral_limit:
                    continue

                # Do not walk beyond the opposite corner of this edge.
                progress = float(np.dot(point - start, unit))
                if target_distance is not None:
                    if progress > target_distance + 0.25 * cell_size:
                        continue

                # Prefer one-cell progress; penalize sideways drift strongly.
                score = abs(forward - cell_size) + 2.0 * lateral
                if score < best_score:
                    best_score = score
                    best_point = point

            if best_point is None:
                break

            key = tuple(np.round(best_point, 3))
            used.add(key)
            chain.append(tuple(float(v) for v in best_point))

        return chain

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


def select_outer_corners(corner_points, img_shape):
    """
    Select the four outer Sudoku corners and verify all four boundary chains.

    Each side is checked independently from one seed corner to the next.
    The expected cell spacing is derived from that side's length / 9, so
    rotation and image resolution do not affect the chain step estimate.

    Returns [(TL), (TR), (BR), (BL)] or None when the boundary evidence
    is insufficient.
    """
    if len(corner_points) < 16:
        return None

    pts_np = np.asarray(corner_points, dtype=np.float32)
    pts_set = [point.copy() for point in pts_np]

    # For the assignment's +/-30 degree rotation range, these extrema provide
    # stable rough seeds. Chain validation below decides whether they are real.
    tl_seed = pts_np[np.argmin(pts_np[:, 0] + pts_np[:, 1])]
    tr_seed = pts_np[np.argmax(pts_np[:, 0] - pts_np[:, 1])]
    br_seed = pts_np[np.argmax(pts_np[:, 0] + pts_np[:, 1])]
    bl_seed = pts_np[np.argmin(pts_np[:, 0] - pts_np[:, 1])]

    seeds = [tl_seed, tr_seed, br_seed, bl_seed]

    # Reject degenerate seed sets before attempting to walk the boundaries.
    if len({tuple(np.round(p, 3)) for p in seeds}) != 4:
        return None

    edge_chains = []
    min_chain_points = 8  # 10 intersections per edge; allow up to 2 missing.

    for index in range(4):
        start = seeds[index]
        end = seeds[(index + 1) % 4]
        edge_vector = end - start
        edge_length = float(np.linalg.norm(edge_vector))

        if edge_length < 9.0:
            return None

        # A standard Sudoku side has 9 cell intervals.
        cell_step = edge_length / 9.0

        chain = CornerDetector.collect_chain(
            start_pt=start,
            direction=edge_vector,
            pts_set=pts_set,
            cell_size=cell_step,
            max_gap_ratio=2.1,
            lateral_tol_ratio=0.22,
            max_steps=9,
            target_distance=edge_length,
        )

        endpoint_error = float(np.linalg.norm(
            np.asarray(chain[-1], dtype=np.float32) - end
        ))
        endpoint_tolerance = max(4.0, 0.65 * cell_step)

        if len(chain) < min_chain_points or endpoint_error > endpoint_tolerance:
            return None

        edge_chains.append(chain)

    # All four boundary chains reached their opposite seed.
    ok, info = CornerDetector.check_quad(seeds, img_shape)
    if not ok:
        return None

    # Keep the same cyclic order convention used by the existing pipeline.
    sorted_pts = CornerDetector.sort_cyclic(seeds)
    return [tuple(float(v) for v in point) for point in sorted_pts]
