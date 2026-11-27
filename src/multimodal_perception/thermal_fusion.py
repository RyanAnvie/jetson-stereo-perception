import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from multimodal_perception.camera import load_calibration, pinhole_camera, rigid_transform
from multimodal_perception.geometry import depth_cloud, thermal_correspondence


def rectified_thermal_camera(raw, settings, frame):
    if raw["distortion_model"] != "equidistant":
        raise ValueError("Expected the source-verified equidistant camera model")
    size = (raw["image_width"], raw["image_height"])
    if list(size) != settings["size"]:
        raise ValueError("Thermal rectification size mismatch")
    K = np.asarray(raw["camera_matrix"]["data"], dtype=float).reshape(3, 3)
    D = np.asarray(raw["distortion_coefficients"]["data"], dtype=float)
    # Reconstruct the producer's output camera. Released images are ALREADY rectified.
    K_rect = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, size, np.asarray(settings["R"], dtype=float),
        balance=settings["balance"], new_size=size, fov_scale=settings["fov_scale"])
    if not np.allclose(K_rect, settings["K_expected"], rtol=0, atol=1e-6):
        raise ValueError("Derived thermal intrinsics differ from the audited profile")
    return pinhole_camera(K_rect, *size, frame)


def load_fusion_calibration(root, profile_path, refinement_path=None):
    root, profile_path = Path(root), Path(profile_path)
    profile = json.loads(profile_path.read_text())
    manifest = json.loads((root / "sample_manifest.json").read_text())
    if manifest["revision"] != profile["dataset_revision"]:
        raise ValueError("Dataset revision is outside the audited thermal profile")
    if manifest["sequence"] != profile["validated_sequence"]:
        raise ValueError("Sequence needs separate thermal registration validation")
    for name, expected in profile["asset_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Thermal profile asset checksum mismatch: {name}")
    rgb_camera, transform = load_calibration(root / "calibration")
    raw = yaml.safe_load((root / "calibration/thermal_left.yaml").read_text())
    camera = rectified_thermal_camera(raw, profile["rectification"], transform["target"])
    evidence = {**profile, "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
                "K_thermal_rectified": camera["K"].tolist(), "K_rgb": rgb_camera["K"].tolist(),
                "rgb_to_thermal_R": transform["R"].tolist(), "rgb_to_thermal_t_m": transform["t"].tolist()}
    if refinement_path is not None:
        refinement_path = Path(refinement_path)
        refinement = json.loads(refinement_path.read_text())
        if (not refinement.get("accepted_for_sequence")
                or refinement["base_profile_sha256"] != evidence["profile_sha256"]
                or refinement["dataset_revision"] != manifest["revision"]
                or refinement["sequence"] != manifest["sequence"]):
            raise ValueError("Scene refinement is not accepted for this calibration and sequence")
        if "thermal_clock_correction_s" in refinement:
            if not refinement.get("timestamp_sha256"):
                raise ValueError("Clock calibration requires timestamp provenance")
            for name, expected in refinement["timestamp_sha256"].items():
                if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
                    raise ValueError(f"Clock calibration timestamp checksum mismatch: {name}")
        transform = rigid_transform(refinement["rgb_to_thermal_R"], refinement["rgb_to_thermal_t_m"],
                                    rgb_camera["frame"], camera["frame"])
        evidence.update(refinement=refinement, refinement_sha256=hashlib.sha256(refinement_path.read_bytes()).hexdigest(),
                        rgb_to_thermal_R=transform["R"].tolist(), rgb_to_thermal_t_m=transform["t"].tolist(),
                        extrinsics="Scene RGB-D / thermal-line bundle adjustment with CAD translation prior",
                        status=refinement.get("status", "scene_bundle_refined_approximate"))
    return {"rgb": rgb_camera, "thermal": camera, "transform": transform, "evidence": evidence}


def fuse_thermal(sample, detections, calibration, stride=2, min_points=20, max_time_error_s=0.032):
    """Associate visible sampled RGB-depth points with processed thermal intensity.

    Occlusion is resolved across the entire sampled frame before selecting boxes.
    A projected box is an association footprint, not a thermal detector prediction.
    """
    if stride < 1 or min_points < 1:
        raise ValueError("Fusion stride and minimum support must be positive")
    empty = {"points": np.empty((0, 3)), "rgb_pixels": np.empty((0, 2), dtype=int),
             "thermal_pixels": np.empty((0, 2), dtype=int), "intensity_u8": np.empty(0, dtype=np.uint8),
             "sampled_depth_points": 0, "status": "invalid_group"}
    times = np.array([sample["timestamp_s"], sample["rgb_timestamp_s"], sample["thermal_timestamp_s"]])
    valid_time = np.isfinite(times).all() and np.max(np.abs(times[1:] - times[0])) <= max_time_error_s
    if not sample["valid"] or sample["ffc"] or not valid_time:
        for detection in detections:
            detection["thermal"] = {"status": "invalid_group", "support_points": 0}
            detection["position"]["thermal_status"] = "invalid_group"
        return empty
    thermal_camera = calibration["thermal"]
    if sample["thermal"].shape != (thermal_camera["height"], thermal_camera["width"]) or sample["thermal"].dtype != np.uint8:
        raise ValueError("Thermal image domain does not match the audited profile")
    points, pixels = depth_cloud(sample["depth"], calibration["rgb"], stride)
    selected, uv = thermal_correspondence(points, calibration["transform"], thermal_camera,
                                          calibration["rgb"]["frame"], verified=True)
    visible_pixels = pixels[selected]
    intensity = sample["thermal"][uv[:, 1], uv[:, 0]]
    for index, detection in enumerate(detections):
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        inside = ((visible_pixels[:, 0] >= x1) & (visible_pixels[:, 0] < x2)
                  & (visible_pixels[:, 1] >= y1) & (visible_pixels[:, 1] < y2))
        count = int(inside.sum())
        association = {"status": "insufficient_support", "support_points": count,
                       "units": "processed_intensity_u8", "frame": thermal_camera["frame"],
                       "method": "RGB bbox / estimated depth / global sampled nearest-Z",
                       "calibration": "source_reconstructed_CAD_nominal"}
        if count >= min_points:
            selected_uv = uv[inside]
            lower, upper = selected_uv.min(axis=0), selected_uv.max(axis=0) + 1
            association.update(status="associated_nominal", bbox_xyxy=np.r_[lower, upper].tolist(),
                               centroid_uv=np.median(selected_uv, axis=0).tolist(),
                               intensity_p10_p50_p90=np.percentile(intensity[inside], [10, 50, 90]).tolist())
        detection.update(id=index + 1, thermal=association)
        detection["position"]["thermal_status"] = association["status"]
    return {"points": points[selected], "rgb_pixels": visible_pixels, "thermal_pixels": uv,
            "intensity_u8": intensity, "sampled_depth_points": len(points),
            "status": calibration.get("evidence", {}).get("status", "source_reconstructed_CAD_nominal")}
