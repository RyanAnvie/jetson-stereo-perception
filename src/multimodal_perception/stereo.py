"""RGB stereo depth; LiDAR is never an input to this estimator."""

import cv2
import numpy as np

from multimodal_perception.camera import transform_points
from multimodal_perception.geometry import project


def create_stereo_matcher(parameters):
    block = parameters["block_size"]
    disparities = parameters["num_disparities"]
    if block < 1 or block % 2 != 1 or disparities <= 0 or disparities % 16:
        raise ValueError("SGBM needs an odd block size and positive disparity range divisible by 16")
    return cv2.StereoSGBM_create(
        minDisparity=parameters["min_disparity"], numDisparities=disparities, blockSize=block,
        P1=8*block*block, P2=32*block*block, preFilterCap=31,
        uniquenessRatio=parameters["uniqueness_ratio"],
        speckleWindowSize=parameters["speckle_window_size"], speckleRange=parameters["speckle_range"],
        disp12MaxDiff=parameters["disp12_max_diff"], mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def disparity_to_depth(disparity, P_left, P_right):
    """Solve both horizontal projection equations, retaining P's fourth column."""
    disparity = np.asarray(disparity, dtype=float)
    u_left = np.broadcast_to(np.arange(disparity.shape[1]), disparity.shape)
    u_right = u_left - disparity
    slope_left = (u_left - P_left[0, 2]) / P_left[0, 0]
    slope_right = (u_right - P_right[0, 2]) / P_right[0, 0]
    intercept_left = (u_left*P_left[2, 3] - P_left[0, 3]) / P_left[0, 0]
    intercept_right = (u_right*P_right[2, 3] - P_right[0, 3]) / P_right[0, 0]
    denominator = slope_left - slope_right
    valid = np.isfinite(disparity) & (disparity > 0) & (denominator > 0)
    z_cam0 = np.full(disparity.shape, np.nan)
    np.divide(intercept_right-intercept_left, denominator, out=z_cam0, where=valid)
    t_left = np.linalg.solve(P_left[:, :3], P_left[:, 3])
    depth = z_cam0 + t_left[2]
    depth[~np.isfinite(depth) | (depth <= 0) | ~valid] = np.nan
    return depth.astype(np.float32)


def estimate_stereo_depth(left, right, matcher, calibration):
    if left.shape != right.shape:
        raise ValueError("Left and right image dimensions differ")
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    # OpenCV returns signed fixed-point disparity with 4 fractional bits.
    disparity = matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
    depth = disparity_to_depth(disparity, calibration["P_left"], calibration["P_right"])
    return depth, disparity


def project_lidar(scan, calibration):
    """Nearest positive Z per rounded RGB pixel; retain original scan indices."""
    points = transform_points(scan[:, :3], calibration["T_cam2_from_velo"], "kitti_velodyne")
    camera = calibration["camera"]
    pixels, visible = project(points, camera)
    indices = np.flatnonzero(visible)
    rounded = np.rint(pixels[indices]).astype(int)
    inside = (rounded[:, 0] >= 0) & (rounded[:, 0] < camera["width"])
    inside &= (rounded[:, 1] >= 0) & (rounded[:, 1] < camera["height"])
    indices, rounded = indices[inside], rounded[inside]
    order = np.argsort(points[indices, 2], kind="stable")
    keys = rounded[:, 1]*camera["width"] + rounded[:, 0]
    _, unique = np.unique(keys[order], return_index=True)
    keep = order[unique]
    return {"scan_indices": indices[keep], "pixels_uv": rounded[keep],
            "points_cam2_m": points[indices[keep]], "subpixel_uv": pixels[indices[keep]]}
