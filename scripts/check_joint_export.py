"""Compare pinned checkpoint, ONNX CPU, and target TensorRT on identical tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def run(stage, directory, report):
    directory = Path(directory)
    inputs = sorted(directory.glob("*.inputs.npz"))
    if not inputs:
        raise ValueError("Run validate_tartan_joint.py first")
    if stage == "torch":
        sys.path.insert(0, str(Path("results/thermal_research/LCAFNet").resolve()))
        import torch
        torch.set_num_threads(4)
        checkpoint = Path("models/weights/LCAFNet_M3FD.pt")
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != "40c041aa51d974f7c7b7f398b45060d420785312e39c33879fd00f1fba49469b":
            raise ValueError("Unexpected upstream checkpoint")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = (saved["ema"] if saved.get("ema") is not None else saved["model"]).float().eval()
        for path in inputs:
            data = np.load(path)
            with torch.no_grad():
                raw = model(torch.from_numpy(data["rgb"]), torch.from_numpy(data["thermal"]))[0].numpy()
            np.save(directory / (path.name.replace(".inputs.npz", ".torch.npy")), raw)
            print(path.name, "checkpoint reference saved", flush=True)
        return
    from multimodal_perception.joint_detection import load_joint_detector, infer_joint, close_joint_detector, decode_joint
    detector = load_joint_detector("models/weights/lcafnet_m3fd.onnx", backend="onnxruntime")
    records = []
    try:
        for path in inputs:
            data = dict(np.load(path))
            onnx, _ = infer_joint(detector, data)
            label = path.name.replace(".inputs.npz", "")
            checkpoint = np.load(directory / f"{label}.torch.npy")
            trt = np.load(directory / f"{label}.trt.npy")
            for name, reference, actual in [("torch_vs_onnx", checkpoint, onnx), ("onnx_vs_trt", onnx, trt)]:
                difference = np.abs(reference-actual)
                expected_boxes = decode_joint(reference, detector["metadata"]["classes"])
                actual_boxes = decode_joint(actual, detector["metadata"]["classes"])
                same_classes = [d["class"] for d in expected_boxes] == [d["class"] for d in actual_boxes]
                box_error = max((np.max(np.abs(np.array(a["bbox_xyxy"])-b["bbox_xyxy"]))
                                 for a, b in zip(expected_boxes, actual_boxes)), default=0.)
                passed = bool(difference.max() < .1 and difference[:, :, 4:].max() < 1e-4
                              and same_classes and box_error < .25)
                records.append({"sample": label, "comparison": name, "raw_absolute_max": float(difference.max()),
                                "probability_absolute_max": float(difference[:, :, 4:].max()),
                                "detections_reference": len(expected_boxes), "detections_actual": len(actual_boxes),
                                "same_ordered_classes": same_classes, "box_absolute_max_px": float(box_error), "passed": passed})
    finally:
        close_joint_detector(detector)
    result = {"checks": records, "all_passed": all(r["passed"] for r in records),
              "thresholds": {"raw_abs": .1, "probability_abs": 1e-4, "decoded_box_abs_px": .25,
                             "classes_and_count": "exact"}}
    Path(report).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    if not result["all_passed"]:
        raise ValueError("Export/runtime numerical checks failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["torch", "compare"])
    parser.add_argument("--directory", default="results/jetson/tartan_joint/validation/reference")
    parser.add_argument("--report", default="reports/jetson/tartan_joint/export_consistency.json")
    args = parser.parse_args()
    run(args.stage, args.directory, args.report)
