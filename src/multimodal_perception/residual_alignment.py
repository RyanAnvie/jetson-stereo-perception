"""Guarded image-domain residual alignment after geometric scene calibration."""
import cv2
import numpy as np


def gradient_magnitude(image):
    blurred = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 2)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    return np.clip(magnitude/max(float(np.percentile(magnitude, 95)), 1.), 0, 1)


def correlation(first, second, valid):
    if np.count_nonzero(valid) < 100:
        return 0.
    a, b = first[valid].astype(float), second[valid].astype(float)
    a -= a.mean()
    b -= b.mean()
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 1e-10 else 0.


def estimate_residual_shift(rgb, thermal, observed, min_score=.3, max_shift=30., min_gain=.01):
    result = {"accepted": False, "delta_uv": [0., 0.], "candidate_delta_uv": [0., 0.],
              "method": "gradient-magnitude ECC translation after scene BA",
              "status": "insufficient_common_support", "physical_pose_correction": False}
    if np.count_nonzero(observed) < .3*observed.size:
        return result
    distance, labels = cv2.distanceTransformWithLabels(255-observed, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    coordinates = np.column_stack(np.nonzero(observed))
    nearest = coordinates[np.maximum(labels-1, 0)]
    # Nearest completion is only used for gradients; masked-out regions never
    # become measured RGB observations or numerical depth samples.
    completed = rgb[nearest[:, :, 0], nearest[:, :, 1]]
    mask = cv2.erode((distance < 3).astype(np.uint8)*255, np.ones((21, 21), np.uint8))
    a = gradient_magnitude(cv2.cvtColor(completed, cv2.COLOR_BGR2GRAY))
    b = gradient_magnitude(thermal)
    baseline = correlation(a, b, mask > 0)
    warp = np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
    try:
        score, warp = cv2.findTransformECC(b, a, warp, cv2.MOTION_TRANSLATION,
                                         (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 80, 1e-5), mask, 5)
    except cv2.error:
        result.update(status="optimizer_failed", correlation_before=baseline)
        return result
    shift = -warp[:, 2]
    corrected = cv2.warpAffine(a, warp, (640, 512), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    corrected_mask = cv2.warpAffine(mask, warp, (640, 512), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    common = (mask > 0) & (corrected_mask > 0)
    before, after = correlation(a, b, common), correlation(corrected, b, common)
    accepted = bool(np.isfinite(shift).all() and np.isfinite(score) and score >= min_score
                    and np.max(np.abs(shift)) <= max_shift and after-before >= min_gain)
    result.update(accepted=accepted, delta_uv=shift.tolist() if accepted else [0., 0.],
                  candidate_delta_uv=shift.tolist(), ecc_score=float(score),
                  correlation_before=before, correlation_after=after, correlation_gain=after-before,
                  status="accepted_image_residual" if accepted else "rejected_by_quality_gate")
    return result


def apply_residual_shift(sample, fusion, rgb, observed, usable, alignment):
    shift = np.asarray(alignment["delta_uv"], dtype=float)
    if not alignment["accepted"]:
        return rgb, observed, usable
    warp = np.array([[1., 0., shift[0]], [0., 1., shift[1]]], dtype=np.float32)
    rgb = cv2.warpAffine(rgb, warp, (640, 512), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
    observed = cv2.warpAffine(observed, warp, (640, 512), flags=cv2.INTER_NEAREST)
    usable = cv2.warpAffine(usable, warp, (640, 512), flags=cv2.INTER_NEAREST)
    integer_shift = np.rint(shift).astype(int)
    uv = fusion["thermal_pixels"] + integer_shift
    valid = (uv[:, 0] >= 0) & (uv[:, 0] < 640) & (uv[:, 1] >= 0) & (uv[:, 1] < 512)
    for key in ["points", "rgb_pixels"]:
        fusion[key] = fusion[key][valid]
    fusion["thermal_pixels"] = uv[valid]
    fusion["intensity_u8"] = sample["thermal"][uv[valid, 1], uv[valid, 0]]
    alignment["geometry_integer_shift_uv"] = integer_shift.tolist()
    alignment["geometry_sampling"] = "nearest thermal pixel; maximum residual-shift quantization 0.5px per axis"
    return rgb, observed, usable
