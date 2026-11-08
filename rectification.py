"""
Stage 3: rectification of the puzzle to a 450x450 square (50 px per cell).

Three models are estimated from the four outer corners (TL, TR, BR, BL):
  A3  affine, exact fit to TL, TR, BL         (cv2.getAffineTransform)
  A4  affine, least squares on all 4 corners  (cv2.estimateAffine2D)
  H4  homography, exact fit to all 4 corners  (cv2.getPerspectiveTransform)
"""
import cv2
import numpy as np

TARGET_SIZE = 450  # 9 cells x 50 px


def target_square(size=TARGET_SIZE):
    """Target corners in the order TL, TR, BR, BL."""
    return np.float32([[0, 0], [size, 0], [size, size], [0, size]])


def apply_affine(M, pts):
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    return pts @ np.asarray(M, np.float64)[:, :2].T + np.asarray(M, np.float64)[:, 2]


def apply_homography(H, pts):
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.asarray(H, np.float64)).reshape(-1, 2)


def _affine_least_squares(src, dst):
    """Exact least-squares affine fit with NumPy (used as a fallback/check)."""
    A = np.hstack([src.astype(np.float64), np.ones((len(src), 1))])
    sol, *_ = np.linalg.lstsq(A, dst.astype(np.float64), rcond=None)
    return sol.T  # 2x3


def _estimate_affine_4pts(src, dst):
    """
    cv2.estimateAffine2D with all four corners. The RANSAC threshold is made
    huge so that all four points are inliers and the final refinement is a
    least-squares fit over all of them.
    """
    M, inliers = cv2.estimateAffine2D(
        src, dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=1e4,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if M is None or inliers is None or int(inliers.sum()) != len(src):
        return _affine_least_squares(src, dst)
    return M.astype(np.float64)


def estimate_models(corners, size=TARGET_SIZE):
    """
    corners: 4 points (TL, TR, BR, BL) in the coordinates of the image that
    will be warped. Returns the three matrices and their diagnostics.
    """
    src = np.asarray(corners, np.float32).reshape(4, 2)
    dst = target_square(size)

    # A3: three point pairs determine an affine map exactly (6 equations,
    # 6 unknowns). The fourth corner (BR) is not used.
    A3 = cv2.getAffineTransform(src[[0, 1, 3]], dst[[0, 1, 3]]).astype(np.float64)
    br_landing = apply_affine(A3, src[2])[0]
    a3_corner_err = float(np.linalg.norm(br_landing - dst[2]))

    # A4: least squares over all four corners (8 equations, 6 unknowns).
    A4 = _estimate_affine_4pts(src, dst)
    residuals = apply_affine(A4, src) - dst
    a4_rms = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))

    # H4: four point pairs determine a homography exactly (8 equations,
    # 8 unknowns).
    H4 = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
    h4_errors = np.linalg.norm(apply_homography(H4, src) - dst, axis=1)

    return {
        "A3": A3,
        "A4": A4,
        "H4": H4,
        "a3_br_landing": br_landing,        # where BR lands in the target
        "a3_corner_err_px": a3_corner_err,  # distance from (450, 450)
        "a4_rms_px": a4_rms,
        "h4_max_corner_err_px": float(h4_errors.max()),
    }


def warp_all(image, models, size=TARGET_SIZE):
    """Returns {"A3": img, "A4": img, "H4": img}, each size x size."""
    dsize = (size, size)
    return {
        "A3": cv2.warpAffine(image, models["A3"], dsize, flags=cv2.INTER_LINEAR),
        "A4": cv2.warpAffine(image, models["A4"], dsize, flags=cv2.INTER_LINEAR),
        "H4": cv2.warpPerspective(image, models["H4"], dsize, flags=cv2.INTER_LINEAR),
    }


def to_original_coordinates(M, scale):
    """
    M maps working-image pixels to the target. The working image is the
    original image multiplied by `scale`, so the matrix that maps ORIGINAL
    pixels to the target is M @ diag(scale, scale, 1).
    """
    M = np.asarray(M, np.float64)
    S = np.diag([scale, scale, 1.0])
    if M.shape == (2, 3):
        return (np.vstack([M, [0.0, 0.0, 1.0]]) @ S)[:2]
    return M @ S
