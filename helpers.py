"""
Small helper functions used by sudoku_scan.py: drawing, puzzle region,
sub-pixel refinement and saving the output files.
"""
import json
import os

import cv2
import numpy as np

import rectification


# ---- saving ---------------------------------------------------------------
def save_images(base_dir, images):
    """images: {"folder/name.png": image}. Creates the folders and saves."""
    for rel_path, img in images.items():
        path = os.path.join(base_dir, rel_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, img)


def save_outputs(outdir, name, result, images):
    """
    Writes the files the grader reads:
      <name>_result.json  the dictionary `result`, saved as text
      <name>_grid.txt     9 lines of 9 digits (only if a grid was read)
      <name>_*.png        every image in `images` ({"_harris.png": image, ...})
    """
    os.makedirs(outdir, exist_ok=True)

    # JSON is just a text format for a Python dictionary: json.dump writes it
    with open(os.path.join(outdir, name + "_result.json"), "w") as file:
        json.dump(result, file, indent=2)

    if result["grid"] is not None:
        lines = ["".join(str(d) for d in row) for row in result["grid"]]
        with open(os.path.join(outdir, name + "_grid.txt"), "w") as file:
            file.write("\n".join(lines) + "\n")

    for suffix, img in images.items():
        cv2.imwrite(os.path.join(outdir, name + suffix), img)


# ---- drawing --------------------------------------------------------------
def draw_points(gray, points, color=(0, 0, 255), radius=6):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x, y in points:
        cv2.circle(vis, (int(x), int(y)), radius, color, -1)
    return vis


def make_harris_image(harris, all_points, candidates, corners):
    """Harris response map. White dots: all candidates, red: candidates in the
    puzzle region, green rings: the four selected corners."""
    response = np.clip(harris, 0, None)
    positive = response[response > 0]
    top = np.percentile(positive, 99.5) if positive.size else 1.0
    normalized = np.clip(response / max(top, 1e-12), 0, 1) ** 0.5
    vis = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    for x, y in all_points:
        cv2.circle(vis, (int(x), int(y)), 3, (255, 255, 255), -1)
    for x, y in candidates:
        cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)
    if corners is not None:
        for x, y in corners:
            cv2.circle(vis, (int(round(x)), int(round(y))), 14, (0, 255, 0), 3)
    return vis


def make_debug_image(rect_bgr, grid, conflicts):
    """Rectified image with cell boundaries, the recognised digits (small, top
    left of each cell) and conflicting cells shaded red."""
    size = rectification.TARGET_SIZE
    cell = size // 9
    img = rect_bgr.copy()

    shade = img.copy()
    for r, c in conflicts:
        cv2.rectangle(shade, (c * cell, r * cell), ((c + 1) * cell, (r + 1) * cell),
                      (0, 0, 255), -1)
    img = cv2.addWeighted(shade, 0.35, img, 0.65, 0)

    for i in range(10):
        thickness = 2 if i % 3 == 0 else 1
        cv2.line(img, (i * cell, 0), (i * cell, size), (255, 0, 0), thickness)
        cv2.line(img, (0, i * cell), (size, i * cell), (255, 0, 0), thickness)

    bad = {tuple(c) for c in conflicts}
    for r in range(9):
        for c in range(9):
            if grid[r][c]:
                color = (0, 0, 200) if (r, c) in bad else (0, 140, 0)
                cv2.putText(img, str(grid[r][c]), (c * cell + 3, r * cell + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


# ---- puzzle region and corner refinement ----------------------------------
def make_region_mask(grid_mask, margin_ratio):
    """
    Region of the puzzle: the largest connected piece of the line mask (the
    grid), grown a little. Printed labels ("9x9", "zor") and neighbouring
    tables are separate pieces, so they stay outside. It is only a region;
    the corner coordinates still come from Harris candidates.
    """
    h, w = grid_mask.shape[:2]
    closed = cv2.morphologyEx(grid_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    region = np.zeros((h, w), np.uint8)
    if n <= 1:
        return region

    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    region[labels == biggest] = 255

    m = max(3, int(round(min(h, w) * margin_ratio)))
    return cv2.dilate(region, np.ones((2 * m + 1, 2 * m + 1), np.uint8))


def filter_by_region(points, region):
    """Keeps only the points that lie inside the region mask."""
    h, w = region.shape[:2]
    return [(x, y) for x, y in points
            if 0 <= int(x) < w and 0 <= int(y) < h and region[int(y), int(x)] > 0]


def refine_corners(gray, corners, win_ratio, max_shift_ratio):
    """
    cornerSubPix on the four selected corners only. A refinement that moves a
    corner more than max_shift_ratio * short side is ignored.
    """
    h, w = gray.shape
    win = max(3, int(round(min(h, w) * win_ratio)))
    max_shift = max_shift_ratio * min(h, w)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)

    refined = []
    for x, y in corners:
        point = np.array([[[x, y]]], np.float32)
        inside = win + 2 <= x < w - win - 2 and win + 2 <= y < h - win - 2
        if inside:
            moved = cv2.cornerSubPix(gray, point.copy(), (win, win), (-1, -1), criteria)
            if np.linalg.norm(moved[0, 0] - point[0, 0]) <= max_shift:
                point = moved
        refined.append(point[0, 0])
    return np.array(refined, np.float32)
