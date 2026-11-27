"""Reproducible image alignment and input-intervention evidence on the target."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from multimodal_perception.dataset import open_dataset, load_frame
from multimodal_perception.thermal_fusion import load_fusion_calibration, fuse_thermal
from multimodal_perception.joint_detection import (
    registered_rgb, preprocess_joint, load_joint_detector, infer_joint, decode_joint, close_joint_detector)
from multimodal_perception.residual_alignment import estimate_residual_shift, apply_residual_shift


def prepare(sample, calibration, config):
    fusion = fuse_thermal(sample, [], calibration, stride=1)
    rgb, observed, usable = registered_rgb(sample, fusion)
    alignment = {"accepted": False, "delta_uv": [0., 0.], "status": "disabled"}
    if config.get("residual_alignment", False):
        alignment = estimate_residual_shift(rgb, sample["thermal"], observed,
                                            config.get("alignment_min_score", .3), config.get("alignment_max_shift_px", 30.),
                                            config.get("alignment_min_gain", .01))
    shifted, _, _ = apply_residual_shift(sample, fusion, rgb, observed, usable, alignment)
    return rgb, shifted, alignment


def labeled(image, text):
    canvas = cv2.copyMakeBorder(image, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=(22, 26, 31))
    cv2.putText(canvas, text, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas


def detection_image(image, detections):
    result = image.copy()
    for d in detections:
        a, b, c, e = np.rint(d["bbox_xyxy"]).astype(int)
        cv2.rectangle(result, (a, b), (c, e), (70, 255, 80), 2)
        cv2.putText(result, f"{d['class']} {d['score']:.2f}", (max(2, a), max(15, b-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, (70, 255, 80), 1, cv2.LINE_AA)
    return result


def validate(config_path, output, full_outputs, sample_root=None):
    config = yaml.safe_load(Path(config_path).read_text())
    if config.get("temporal_compensation", False):
        raise ValueError("Use validate_fixed_sequence.py for clock-compensated registration; this legacy validator does not model exposure-time poses")
    cv2.setNumThreads(config["opencv_threads"])
    dataset = open_dataset(config["dataset_root"])
    nominal = load_fusion_calibration(config["dataset_root"], config["thermal_profile"])
    calibration = load_fusion_calibration(config["dataset_root"], config["thermal_profile"], config["thermal_refinement"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    artifact = Path("results/jetson/tartan_joint/validation")
    artifact.mkdir(parents=True, exist_ok=True)
    reference = artifact / "reference"
    reference.mkdir(exist_ok=True)
    refinement = calibration["evidence"]["refinement"]
    alignments = {}
    for index in refinement["heldout_frames"]:
        sample = load_frame(dataset, index)
        raw, shifted, alignment = prepare(sample, calibration, config)
        nominal_rgb, _, _ = registered_rgb(sample, fuse_thermal(sample, [], nominal, stride=1))
        heat = cv2.applyColorMap(sample["thermal"], cv2.COLORMAP_INFERNO)
        panels = [labeled(cv2.addWeighted(rgb, .55, heat, .45, 0), title)
                  for rgb, title in [(nominal_rgb, "CAD nominal"), (raw, "Scene BA"), (shifted, "Scene BA + guarded image residual")]]
        cv2.imwrite(str(artifact / f"alignment_{index:04d}.jpg"), np.hstack(panels))
        alignments[index] = alignment
    observations = []
    for observation in refinement["heldout_observations"]:
        alignment = alignments[observation["index"]]
        after_x = observation["after_x"] + alignment["delta_uv"][0]
        observations.append({**observation, "ba_plus_residual_x": after_x,
                             "ba_plus_residual_abs_horizontal_px": abs(after_x-observation["thermal_x"]),
                             "residual_accepted": alignment["accepted"]})
    error_summary = {}
    for name, field in [("nominal", "before_abs_horizontal_px"), ("scene_ba", "after_abs_horizontal_px"),
                        ("scene_ba_plus_residual", "ba_plus_residual_abs_horizontal_px")]:
        values = [o[field] for o in observations]
        error_summary[name] = {"mean_px": float(np.mean(values)), "max_px": float(np.max(values))}
    alignment_report = {"errors": error_summary, "observations": observations, "per_frame_alignment": alignments,
                        "scope": "6 horizontal edge observations in 3 frames, excluded from BA fitting; about +/-4px annotation uncertainty. Development validation, not independent calibration certification, full 2D error, or cross-sequence accuracy.",
                        "image_residual_changes_physical_pose": False}
    (output / "alignment_validation.json").write_text(json.dumps(alignment_report, indent=2)+"\n")

    all_outputs = json.loads(Path(full_outputs).read_text())
    uniform = list(range(0, 296, 25)) + [295]
    positive = [o["index"] for o in all_outputs if o["detections"]]
    selected = sorted(set(uniform+positive))
    detector = load_joint_detector(config["model"], config["engine"], config["backend"], config["onnx_threads"])
    evidence = []
    try:
        pairs = []
        for index in selected:
            sample = load_frame(dataset, index)
            _, rgb, _ = prepare(sample, calibration, config)
            pairs.append((f"tartan_{index:04d}", rgb, sample["thermal"], "uniform" if index in uniform else "all_positive_frames"))
        if sample_root:
            sample_root = Path(sample_root)
            rgb_path, heat_path = sample_root / "vi/M3FD_00471.png", sample_root / "ir/M3FD_00471.png"
            rgb = cv2.imread(str(rgb_path))
            heat = cv2.imread(str(heat_path), cv2.IMREAD_GRAYSCALE)
            # Keep the example's aspect ratio: 1024x768 -> 640x480 plus 16px borders.
            if rgb is None or heat is None or rgb.shape[:2] != (768, 1024) or heat.shape != (768, 1024):
                raise ValueError("Unexpected official M3FD sample dimensions")
            rgb = cv2.copyMakeBorder(cv2.resize(rgb, (640, 480)), 16, 16, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            heat = cv2.copyMakeBorder(cv2.resize(heat, (640, 480)), 16, 16, 0, 0, cv2.BORDER_CONSTANT, value=114)
            pairs.append(("M3FD_00471", rgb, heat, "separate_positive_smoke_example_not_Tartan"))
            (output / "sample_provenance.json").write_text(json.dumps({
                "repository": "https://github.com/dlut-dimt/TarDAL", "commit": "6a9edd744b44fc03344fe8fb0fd930f5df47b00b",
                "path": "assets/sample/s1", "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [rgb_path, heat_path]},
                "scope": "Known paired example; may overlap pretrained model training data; not accuracy validation."}, indent=2)+"\n")
        for label, rgb, heat, selection in pairs:
            raw_outputs, detections, canvases = {}, {}, []
            for mode in ["joint", "thermal_flat", "rgb_flat"]:
                tensors = preprocess_joint(rgb, heat, mode)
                raw, _ = infer_joint(detector, tensors)
                raw_outputs[mode] = raw
                detections[mode] = decode_joint(raw, detector["metadata"]["classes"], config["score_threshold"], config["nms_iou"])
                canvases.append(labeled(detection_image(cv2.addWeighted(rgb, .5, cv2.cvtColor(heat, cv2.COLOR_GRAY2BGR), .5, 0), detections[mode]),
                                       f"{label} / {mode} / {len(detections[mode])} detections"))
                if mode == "joint" and label in ["tartan_0013", "tartan_0114", "M3FD_00471"]:
                    np.savez_compressed(reference / f"{label}.inputs.npz", **tensors)
                    np.save(reference / f"{label}.trt.npy", raw)
            changes = {}
            for mode in ["thermal_flat", "rgb_flat"]:
                difference = np.abs(raw_outputs[mode][:, :, 4:]-raw_outputs["joint"][:, :, 4:])
                changes[mode] = {"probability_abs_max": float(difference.max()), "probability_abs_mean": float(difference.mean())}
            evidence.append({"sample": label, "selection": selection, "detections": detections, "input_intervention": changes})
            if any(detections.values()) or label == "M3FD_00471":
                cv2.imwrite(str(artifact / f"ablation_{label}.jpg"), np.hstack(canvases))
            print(label, {k: len(v) for k, v in detections.items()}, flush=True)
    finally:
        close_joint_detector(detector)
    report = {"selection": "13 uniform Tartan frames plus every frame with joint detections; optional official example reported separately",
              "uniform_indices": uniform, "positive_indices": positive, "threshold": config["score_threshold"],
              "scope": "Mean-flat input intervention with identical registration; tests whether each input affects recognition. Not a comparison with separately trained unimodal models or detection accuracy.",
              "records": evidence}
    (output / "modality_ablation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(error_summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tartan_joint.yaml")
    parser.add_argument("--output", default="reports/jetson/tartan_joint")
    parser.add_argument("--full-outputs", default="reports/jetson/tartan_joint/joint_fp32.outputs.json")
    parser.add_argument("--sample-root")
    args = parser.parse_args()
    validate(args.config, args.output, args.full_outputs, args.sample_root)
