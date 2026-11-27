"""RGB-only rigid scene motion for a constant sensor-clock correction.

The inter-camera extrinsic stays fixed. Motion maps source RGB-depth points
into the RGB camera coordinate system at the corrected thermal exposure time.
"""
import cv2
import numpy as np

from multimodal_perception.camera import rigid_transform
from multimodal_perception.geometry import backproject


def estimate_rgb_motion(first, second, camera):
    cv2.setRNGSeed(42)
    gray = cv2.cvtColor(first["rgb"], cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(second["rgb"], cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 700, .01, 10)
    result = {"accepted": False, "status": "insufficient_tracks", "source_index": first["index"],
              "target_index": second["index"], "dt_s": second["rgb_timestamp_s"]-first["rgb_timestamp_s"]}
    if corners is None or len(corners) < 20:
        return result
    uv = corners[:, 0]
    xy = np.rint(uv).astype(int)
    depth = first["depth"][xy[:, 1], xy[:, 0]]
    points, good = backproject(uv, depth, camera)
    good &= (depth >= .3) & (depth <= 30)
    tracked, ok, _ = cv2.calcOpticalFlowPyrLK(gray, next_gray, corners, None, winSize=(31, 31), maxLevel=4)
    reverse, back, _ = cv2.calcOpticalFlowPyrLK(next_gray, gray, tracked, None, winSize=(31, 31), maxLevel=4)
    good &= ok.ravel().astype(bool) & back.ravel().astype(bool)
    good &= np.linalg.norm(reverse[:, 0]-uv, axis=1) < .7
    good &= (tracked[:, 0, 0] > 4) & (tracked[:, 0, 0] < gray.shape[1]-4)
    good &= (tracked[:, 0, 1] > 4) & (tracked[:, 0, 1] < gray.shape[0]-4)
    ids = np.flatnonzero(good)
    result["tracked_depth_points"] = len(ids)
    if len(ids) < 20:
        return result
    success, rotation, translation, inliers = cv2.solvePnPRansac(points[ids], tracked[ids, 0], camera["K"], None,
                                                                iterationsCount=200, reprojectionError=1.8,
                                                                confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    if not success or inliers is None or len(inliers) < 20:
        return result
    selected = ids[inliers.ravel()]
    rotation, translation = cv2.solvePnPRefineLM(points[selected], tracked[selected, 0], camera["K"], None, rotation, translation)
    projected, _ = cv2.projectPoints(points[selected], rotation, translation, camera["K"], None)
    errors = np.linalg.norm(projected[:, 0]-tracked[selected, 0], axis=1)
    spread = cv2.contourArea(cv2.convexHull(uv[selected])) / gray.size
    accepted = bool(len(selected)/len(ids) >= .35 and np.percentile(errors, 95) < 2.0 and spread > .04
                    and np.linalg.norm(rotation) < .6 and np.isfinite(translation).all())
    result.update(accepted=accepted, status="accepted_rgb_rigid_motion" if accepted else "rejected_motion_quality",
                  rvec_rad=rotation.ravel().tolist(), translation_m=translation.ravel().tolist(),
                  inliers=len(selected), inlier_fraction=len(selected)/len(ids), image_hull_fraction=spread,
                  reprojection_p50_p95_px=np.percentile(errors, [50, 95]).tolist())
    return result


def interpolate_rgb_motion(motion, time_from_rgb_s):
    if not motion["accepted"]:
        raise ValueError("RGB motion did not pass the quality gate")
    alpha = time_from_rgb_s / motion["dt_s"]
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Corrected exposure is outside the motion bracket")
    R = cv2.Rodrigues(np.asarray(motion["rvec_rad"])*alpha)[0]
    t = np.asarray(motion["translation_m"])*alpha
    return R, t


def exposure_transform(calibration, motion, time_from_rgb_s):
    """Compose fixed thermal-from-RGB extrinsics with interpolated rig motion."""
    R, t = interpolate_rgb_motion(motion, time_from_rgb_s)
    fixed = calibration["transform"]
    return rigid_transform(fixed["R"] @ R, fixed["R"] @ t + fixed["t"], fixed["source"], fixed["target"])


def load_exposure_sample(dataset, index, calibration, enabled, loader=None):
    from multimodal_perception.dataset import load_frame
    loader = load_frame if loader is None else loader
    sample = loader(dataset, index).copy()
    if not enabled:
        return sample
    correction = calibration["evidence"].get("refinement", {}).get("thermal_clock_correction_s")
    if correction is None:
        raise ValueError("Temporal compensation requires sequence-validated clock calibration")
    exposure = sample["thermal_timestamp_s"]+correction
    if exposure < sample["rgb_timestamp_s"]:
        raise ValueError("This release only supports forward RGB exposure brackets")
    for adjacent in range(index+1, min(index+4, len(dataset["target"]))):
        if dataset["zed_left_rect"]["actual"][adjacent] >= exposure:
            sample["rgb_motion_neighbor"] = loader(dataset, adjacent)
            break
    return sample


def calibration_at_exposure(sample, calibration, enabled):
    if not enabled:
        return calibration, {"enabled": False, "compensated": False, "status": "disabled",
                             "fixed_extrinsics": True, "thermal_clock_correction_s": 0.}
    correction = calibration["evidence"]["refinement"]["thermal_clock_correction_s"]
    exposure = sample["thermal_timestamp_s"]+correction
    duration = exposure-sample["rgb_timestamp_s"]
    record = {"enabled": True, "compensated": False, "status": "missing_rgb_bracket", "fixed_extrinsics": True,
              "thermal_clock_correction_s": correction, "effective_thermal_timestamp_s": exposure,
              "exposure_after_rgb_s": duration,
              "assumption": "Rigid scene; RGB-only PnP plus rotation/translation interpolation; no cross-modal image shift fitting."}
    if "rgb_motion_neighbor" not in sample:
        return calibration, record
    motion = estimate_rgb_motion(sample, sample["rgb_motion_neighbor"], calibration["rgb"])
    record["rgb_motion"] = motion
    if not motion["accepted"]:
        record["status"] = "uncompensated_motion_unreliable"
        return calibration, record
    transform = exposure_transform(calibration, motion, duration)
    record.update(compensated=True, status="fixed_clock_and_rgb_rig_motion", motion_alpha=duration/motion["dt_s"],
                  exposure_transform_R=transform["R"].tolist(), exposure_transform_t_m=transform["t"].tolist())
    return {**calibration, "transform": transform}, record
