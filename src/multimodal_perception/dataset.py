import json
from pathlib import Path

import cv2
import numpy as np
import yaml


def frame_files(directory, suffix):
    mapping = {}
    for path in Path(directory).glob("*" + suffix):
        if not path.stem.isdigit():
            raise ValueError(f"Unexpected frame name: {path}")
        index = int(path.stem)
        if index in mapping:
            raise ValueError(f"Duplicate frame index: {index}")
        mapping[index] = path
    return mapping


def match_timestamps(target, actual, errors, max_error_s):
    """The release maps line i to frame i; verify that mapping against sidecars."""
    target = np.asarray(target, dtype=float)
    actual = np.asarray(actual, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if len(target) != len(actual) or len(target) != len(errors):
        raise ValueError("Timestamp sidecar length mismatch")
    if not np.isfinite(target).all() or np.any(np.diff(target) <= 0):
        raise ValueError("Target timestamps must be finite and strictly increasing")
    if max_error_s < 0:
        raise ValueError("Time tolerance must be nonnegative")
    finite = np.isfinite(actual) & np.isfinite(errors) & (actual > 0) & (errors >= 0)
    delta = np.abs(actual - target)
    if not np.allclose(delta[finite], errors[finite], atol=2e-6, rtol=0):
        raise ValueError("Timestamp/error sidecars contradict the master clock")
    valid = finite & (delta <= max_error_s)
    return valid, delta


def open_dataset(root, max_time_error_s=0.032):
    root = Path(root)
    manifest = json.loads((root / "sample_manifest.json").read_text())
    sequence = root / manifest["sequence"]
    target = np.atleast_1d(np.loadtxt(sequence / "target_timestamps.txt"))
    metadata = {"root": root, "sequence": sequence, "manifest": manifest, "target": target}
    valid = np.ones(len(target), dtype=bool)
    reasons = [[] for _ in target]
    for name in ["zed_left_rect", "thermal_left_rect_8"]:
        folder = sequence / name
        actual = np.atleast_1d(np.loadtxt(folder / "timestamps.txt"))
        errors = np.atleast_1d(np.loadtxt(folder / "errors.txt"))
        good, delta = match_timestamps(target, actual, errors, max_time_error_s)
        metadata[name] = {"files": frame_files(folder, ".png"), "actual": actual, "delta": delta,
                          "frame": yaml.safe_load((folder / "frames.yaml").read_text())["frame_id"]}
        for i in np.flatnonzero(~good):
            reasons[i].append(name + "_timestamp_invalid")
        valid &= good
    expected_frames = {"zed_left_rect": "zed_left_camera_optical_frame", "thermal_left_rect_8": "thermal_left/optical_frame"}
    for name, expected in expected_frames.items():
        if metadata[name]["frame"] != expected:
            raise ValueError(f"Unexpected optical frame for {name}")
    metadata["stereo_depth"] = {"files": frame_files(sequence / "stereo_depth", ".npy")}
    ffc = np.atleast_1d(np.loadtxt(sequence / "thermal_left_ffc/data.txt"))
    ffc_errors = np.atleast_1d(np.loadtxt(sequence / "thermal_left_ffc/errors.txt"))
    if len(ffc) != len(target) or len(ffc_errors) != len(target) or not np.isin(ffc, [0, 1]).all():
        raise ValueError("Invalid or incomplete FFC sidecars")
    ffc_good = (ffc == 0) & np.isfinite(ffc_errors) & (ffc_errors >= 0) & (ffc_errors <= max_time_error_s)
    for i in np.flatnonzero(~ffc_good):
        reasons[i].append("ffc_or_ffc_timestamp_invalid")
    valid &= ffc_good
    for name in ["zed_left_rect", "thermal_left_rect_8", "stereo_depth"]:
        files = metadata[name]["files"]
        if any(i < 0 or i >= len(target) for i in files):
            raise ValueError(f"Out-of-range frame index in {name}")
        for i in range(len(target)):
            if i not in files:
                valid[i] = False
                reasons[i].append(name + "_missing")
    metadata.update(valid=valid, reasons=reasons, ffc=ffc, ffc_errors=ffc_errors,
                    max_time_error_s=max_time_error_s)
    return metadata


def load_frame(dataset, index):
    rgb = cv2.imread(str(dataset["zed_left_rect"]["files"][index]), cv2.IMREAD_COLOR)
    thermal = cv2.imread(str(dataset["thermal_left_rect_8"]["files"][index]), cv2.IMREAD_UNCHANGED)
    depth = np.load(dataset["stereo_depth"]["files"][index], allow_pickle=False)
    if rgb is None or rgb.shape != (540, 960, 3):
        raise ValueError(f"RGB decode/shape failure at {index}")
    if thermal is None or thermal.shape != (512, 640) or thermal.dtype != np.uint8:
        raise ValueError(f"Thermal decode/shape failure at {index}")
    if depth.shape != (540, 960) or depth.dtype != np.float32:
        raise ValueError(f"Depth shape/dtype failure at {index}")
    return {"index": index, "rgb": rgb, "thermal": thermal, "depth": depth,
            "sequence": dataset["manifest"]["sequence"],
            "timestamp_s": float(dataset["target"][index]),
            "rgb_timestamp_s": float(dataset["zed_left_rect"]["actual"][index]),
            "thermal_timestamp_s": float(dataset["thermal_left_rect_8"]["actual"][index]),
            "ffc": bool(dataset["ffc"][index]), "valid": bool(dataset["valid"][index]),
            "reasons": dataset["reasons"][index], "units": "m"}


def audit_dataset(dataset):
    fractions = []
    nan_fractions = []
    depth_min = float("inf")
    depth_max = -float("inf")
    positive_samples = []
    checked = 0
    for i in range(len(dataset["target"])):
        if any(i not in dataset[name]["files"] for name in ["zed_left_rect", "thermal_left_rect_8", "stereo_depth"]):
            continue
        sample = load_frame(dataset, i)
        depth = sample["depth"]
        positive = np.isfinite(depth) & (depth > 0)
        fractions.append(float(positive.mean()))
        nan_fractions.append(float(np.isnan(depth).mean()))
        finite = depth[np.isfinite(depth)]
        if finite.size:
            depth_min = min(depth_min, float(finite.min()))
            depth_max = max(depth_max, float(finite.max()))
        positive_samples.extend(depth[positive][::100].tolist())
        checked += 1
    delta = dataset["thermal_left_rect_8"]["delta"]
    finite_delta = delta[np.isfinite(delta)]
    report = {"revision": dataset["manifest"]["revision"], "sequence": dataset["manifest"]["sequence"],
              "target_frames": len(dataset["target"]), "decoded_groups": checked,
              "valid_matched_groups": int(dataset["valid"].sum()), "ffc_count": int(dataset["ffc"].sum()),
              "max_time_error_s": dataset["max_time_error_s"],
              "nominal_release_hz": 10,
              "actual_target_interval_ms_p50_p95": np.percentile(np.diff(dataset["target"])*1000, [50, 95]).tolist(),
              "actual_target_span_s": float(dataset["target"][-1]-dataset["target"][0]),
              "actual_target_rate_hz": float((len(dataset["target"])-1)/(dataset["target"][-1]-dataset["target"][0])),
              "thermal_target_offset_ms_p50_p95": np.percentile(finite_delta * 1000, [50, 95]).tolist(),
              "rgb_thermal_offset_ms_p50_p95": np.percentile(np.abs(dataset["zed_left_rect"]["actual"] - dataset["thermal_left_rect_8"]["actual"]) * 1000, [50, 95]).tolist(),
              "depth_dtype": "float32", "depth_units": "m", "depth_source": "FoundationStereo estimate, not independent ground truth",
              "depth_finite_range_m": [depth_min, depth_max] if checked else None,
              "positive_depth_sample_percentiles_1_50_99_m": np.percentile(positive_samples, [1, 50, 99]).tolist() if positive_samples else None,
              "mean_valid_depth_fraction": float(np.mean(fractions)) if fractions else None,
              "mean_nan_fraction": float(np.mean(nan_fractions)) if nan_fractions else None,
              "shapes": {"rgb": [540, 960, 3], "thermal": [512, 640], "depth": [540, 960]},
              "rejected": [{"index": i, "reasons": reasons} for i, reasons in enumerate(dataset["reasons"]) if reasons]}
    return report
