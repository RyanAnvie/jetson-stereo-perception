"""Whole-sequence fixed-rig counterfactuals, with no cross-modal frame fitting."""
import json
from pathlib import Path

import cv2
import numpy as np

from multimodal_perception.dataset import open_dataset
from multimodal_perception.thermal_fusion import load_fusion_calibration, fuse_thermal
from multimodal_perception.joint_detection import registered_rgb
from multimodal_perception.rig_motion import load_exposure_sample, calibration_at_exposure, exposure_transform
from multimodal_perception.residual_alignment import gradient_magnitude, correlation


def gradients(rgb, observed):
    distance, labels = cv2.distanceTransformWithLabels(255-observed, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    coordinates = np.column_stack(np.nonzero(observed))
    nearest = coordinates[np.maximum(labels-1, 0)]
    complete = rgb[nearest[:, :, 0], nearest[:, :, 1]]
    gray = cv2.cvtColor(complete, cv2.COLOR_BGR2GRAY)
    mask = cv2.erode((distance < 3).astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
    return gradient_magnitude(gray), mask


def validate():
    cv2.setNumThreads(4)
    data = open_dataset("data/tartan")
    old = load_fusion_calibration("data/tartan", "configs/tartan_thermal_profile.json", "configs/tartan_scene_refinement.json")
    new = load_fusion_calibration("data/tartan", "configs/tartan_thermal_profile.json", "configs/tartan_fixed_refinement.json")
    folder = Path("reports/jetson/tartan_fixed")
    images = Path("results/jetson/tartan_fixed/validation"); images.mkdir(parents=True, exist_ok=True)
    records = []
    for index in np.flatnonzero(data["valid"]).tolist():
        sample = load_exposure_sample(data, index, new, True)
        exposure, motion = calibration_at_exposure(sample, new, True)
        variants = {"old_fixed_no_ecc": old, "refit_fixed_no_time_compensation": new,
                    "refit_fixed_with_time_compensation": exposure}
        if motion["compensated"]:
            variants["old_fixed_with_time_compensation"] = {**old, "transform": exposure_transform(old, motion["rgb_motion"], motion["exposure_after_rgb_s"])}
        warped, masks, fields = {}, {}, {}
        for name, calibration in variants.items():
            fusion = fuse_thermal(sample, [], calibration, stride=1)
            rgb, observed, _ = registered_rgb(sample, fusion)
            warped[name] = rgb
            fields[name], masks[name] = gradients(rgb, observed)
        common = np.logical_and.reduce(list(masks.values()))
        thermal_gradient = gradient_magnitude(sample["thermal"])
        scores = {key: correlation(value, thermal_gradient, common) for key, value in fields.items()}
        record = {"index": index, "common_fraction": float(common.mean()), "gradient_magnitude_correlation": scores,
                  "motion": motion, "shared_extrinsics": True, "per_frame_cross_modal_fitting": False}
        records.append(record)
        if index % 20 == 0 or index in [100, 150, 250, 290, 295]:
            panels = []
            for name in ["old_fixed_no_ecc", "refit_fixed_with_time_compensation"]:
                heat = cv2.applyColorMap(sample["thermal"], cv2.COLORMAP_INFERNO)
                panel = cv2.addWeighted(warped[name], .55, heat, .45, 0)
                panel = cv2.copyMakeBorder(panel, 35, 0, 0, 0, cv2.BORDER_CONSTANT)
                cv2.putText(panel, f"{index} {name} / corr {scores[name]:.3f}", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1)
                panels.append(panel)
            cv2.imwrite(str(images/f"fixed_{index:04d}.jpg"), np.hstack(panels))
        if index % 25 == 0: print(index, scores, motion["status"], flush=True)
    common_rows = [r for r in records if r["motion"]["compensated"]]
    aggregate = {}
    for name, rows in [("all_valid_frames", records), ("motion_supported_frames", common_rows)]:
        metric = {}
        for key in ["old_fixed_no_ecc", "refit_fixed_no_time_compensation", "refit_fixed_with_time_compensation", "old_fixed_with_time_compensation"]:
            values = [r["gradient_magnitude_correlation"][key] for r in rows if key in r["gradient_magnitude_correlation"]]
            metric[key] = {"frames": len(values), "mean": float(np.mean(values)), "p10_p50_p90": np.percentile(values, [10, 50, 90]).tolist()}
        aggregate[name] = metric
    gained = [r["gradient_magnitude_correlation"]["refit_fixed_with_time_compensation"]-r["gradient_magnitude_correlation"]["old_fixed_no_ecc"] for r in records]
    report = {"valid_frames": len(records), "motion_supported_frames": len(common_rows),
              "uncompensated_frames": [r["index"] for r in records if not r["motion"]["compensated"]],
              "frames_with_higher_correlation": int((np.array(gained) > 0).sum()), "aggregate": aggregate,
              "scope": "Whole-frame structural image proxy on identical eroded support for each ablation. Thermal/RGB appearance differs; correlation is not a pixel ground-truth error or detection accuracy. No ECC, per-frame cross-modal optimization, or per-frame extrinsic fitting.",
              "records": records}
    (folder/"fixed_sequence_validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    validate()
