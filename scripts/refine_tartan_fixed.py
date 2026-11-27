"""Fit one rigid extrinsic after independently measuring a constant clock offset."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from multimodal_perception.dataset import open_dataset, load_frame
from multimodal_perception.thermal_fusion import load_fusion_calibration
from multimodal_perception.geometry import depth_cloud, thermal_correspondence
from multimodal_perception.scene_calibration import collect_scene_windows, fit_scene_bundle, project_pose
from multimodal_perception.rig_motion import interpolate_rgb_motion


def refine():
    cv2.setNumThreads(4); cv2.setRNGSeed(42)
    folder = Path("reports/jetson/tartan_fixed")
    dataset = open_dataset("data/tartan")
    calibration = load_fusion_calibration("data/tartan", "configs/tartan_thermal_profile.json")
    annotations_path = Path("configs/tartan_scene_lines.json")
    annotations = json.loads(annotations_path.read_text())
    lag = .070
    windows = collect_scene_windows(dataset, calibration, annotations, thermal_clock_correction_s=lag)
    pose, optimization = fit_scene_bundle(windows, calibration, annotations["annotation_sigma_px"])
    motions = json.loads((folder/"rgb_motion.json").read_text())
    old = json.loads(Path("configs/tartan_scene_refinement.json").read_text())
    records = []
    for observation in old["heldout_observations"]:
        index = observation["index"]
        sample = load_frame(dataset, index)
        points, _ = depth_cloud(sample["depth"], calibration["rgb"], 1)
        selected, uv = thermal_correspondence(points, calibration["transform"], calibration["thermal"], calibration["rgb"]["frame"], True)
        _, nearest = cKDTree(uv).query([observation["projected_rgb_x"], observation["row_y"]])
        point = points[selected[nearest]][None]
        R, t = interpolate_rgb_motion(motions[index], lag+sample["thermal_timestamp_s"]-sample["rgb_timestamp_s"])
        predicted = project_pose(point@R.T+t, pose, calibration["thermal"]["K"])[0]
        records.append({**observation, "fixed_clock_x": float(predicted[0]),
                        "fixed_clock_abs_horizontal_px": float(abs(predicted[0]-observation["thermal_x"]))})
    errors = np.array([r["fixed_clock_abs_horizontal_px"] for r in records])
    previous = np.array([r["after_abs_horizontal_px"] for r in records])
    accepted = bool(optimization["solver_success"] and errors.mean() < previous.mean() and errors.max() < previous.max())
    record = {"method": "One shared rigid RGB/thermal extrinsic from scene BA with fixed clock offset and interpolated RGB rig poses",
              "accepted_for_sequence": accepted, "base_profile_sha256": calibration["evidence"]["profile_sha256"],
              "dataset_revision": dataset["manifest"]["revision"], "sequence": dataset["manifest"]["sequence"],
              "annotations_sha256": hashlib.sha256(annotations_path.read_bytes()).hexdigest(),
              "training_frames": annotations["training_frames"], "heldout_frames": annotations["heldout_frames"],
              "rgb_to_thermal_R": cv2.Rodrigues(pose[:3])[0].tolist(), "rgb_to_thermal_t_m": pose[3:].tolist(),
              "rvec_rad": pose[:3].tolist(), "thermal_clock_correction_s": lag,
              "clock_convention": "effective thermal exposure = released thermal timestamp + correction; RGB timestamps unchanged",
              "clock_scope": "Sequence-specific image-motion alignment, not an independently measured hardware clock offset",
              "timestamp_sha256": {str(path.relative_to(Path("data/tartan"))): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in [dataset["sequence"]/"target_timestamps.txt",
                                                dataset["sequence"]/"zed_left_rect/timestamps.txt",
                                                dataset["sequence"]/"thermal_left_rect_8/timestamps.txt"]},
              "heldout_observations": records, "heldout_horizontal_before_mean_max_px": [float(previous.mean()), float(previous.max())],
              "heldout_horizontal_after_mean_max_px": [float(errors.mean()), float(errors.max())],
              "status": "fixed_extrinsics_clock_refined_approximate", "physical_accuracy_certified": False,
              "validation_scope": "Six original manual horizontal edges, +/-4px annotation uncertainty. Full-sequence motion and image proxy diagnostics are reported separately; no per-frame ECC."}
    (folder/"fixed_scene_bundle_adjustment.json").write_text(json.dumps({**record, "bundle_adjustment": optimization}, indent=2)+"\n")
    if accepted:
        Path("configs/tartan_fixed_refinement.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({k: record[k] for k in ["accepted_for_sequence", "rvec_rad", "rgb_to_thermal_t_m", "heldout_horizontal_before_mean_max_px", "heldout_horizontal_after_mean_max_px"]}, indent=2))
    if not accepted:
        raise ValueError("Clock-aware rigid calibration did not pass the manual validation gate")


if __name__ == "__main__":
    refine()
