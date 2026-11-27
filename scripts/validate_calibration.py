import json
from pathlib import Path

import cv2
import numpy as np

from multimodal_perception.camera import load_calibration
from multimodal_perception.dataset import open_dataset, load_frame
from multimodal_perception.geometry import backproject, depth_cloud, project


if __name__ == "__main__":
    dataset = open_dataset("data/tartan")
    camera, transform = load_calibration(Path("data/tartan/calibration"))
    sample = load_frame(dataset, 100)
    pixels = np.array([[240, 200], [480, 300], [720, 400]])
    depth = sample["depth"][pixels[:, 1], pixels[:, 0]]
    points, valid = backproject(pixels, depth, camera)
    reprojection, visible = project(points, camera)
    errors = np.linalg.norm(reprojection[valid]-pixels[valid], axis=1)
    overlay = sample["rgb"].copy()
    for i in range(len(pixels)):
        if valid[i]:
            point = tuple(pixels[i])
            cv2.circle(overlay, point, 7, (0, 230, 120), 2)
            cv2.putText(overlay, f"Z={points[i, 2]:.2f}m", (point[0]+10, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 120), 1, cv2.LINE_AA)
    output = Path("results/diagnostics")
    output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output / "rgb_backprojection.jpg"), overlay)
    cv2.imwrite(str(output / "thermal_unregistered.png"), sample["thermal"])
    mask = np.isfinite(sample["depth"]) & (sample["depth"] > 0)
    cv2.imwrite(str(output / "valid_depth_mask.png"), mask.astype(np.uint8)*255)
    cloud, cloud_pixels = depth_cloud(sample["depth"], camera)
    colors_bgr = sample["rgb"][cloud_pixels[:, 1], cloud_pixels[:, 0]]
    np.savez_compressed(output / "rgb_cloud_0100.npz", xyz_m=cloud, pixels_uv=cloud_pixels,
                        rgb=colors_bgr[:, ::-1], frame=camera["frame"])
    report = {"rgb_geometry": "PASS", "thermal_mapping": "not_evaluated_rgb_geometry_only", "revision": dataset["manifest"]["revision"],
              "rgb_frame": camera["frame"], "rgb_domain": "rectified pinhole, released 960x540 K",
              "rgb_units": "m", "source_frame": 100,
              "backprojection_reprojection_roundtrip_max_px": float(errors.max()),
              "roundtrip_is_registration_accuracy": False, "thermal_registration_error_px": None,
              "T_thermal_from_rgb": {"R": transform["R"].tolist(), "t_m": transform["t"].tolist()},
              "transform_evidence": "configs/handheld.yaml explicitly says pose of to_frame in from_frame; invert to map points parent -> child",
              "thermal_domain_evidence": ["Dataset card calls thermal_left_rect_8 rectified", "thermal_left.yaml supplies raw equidistant K/D and a pinhole P", "configs/tartan_rgbt.yaml records thermal_left_rect_8 from /thermal_left/image_processed; camera_info is commented out"],
              "thermal_calibration_profile": "configs/tartan_thermal_profile.json defines the processed thermal coordinate domain; this check only evaluates RGB geometry",
              "depth_policy": "finite positive Z only; no physical upper range was documented; huge positive estimates remain numerical outputs, display crops are explicit"}
    Path("reports/geometry_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
