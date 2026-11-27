import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from multimodal_perception.camera import pinhole_camera
from multimodal_perception.dataset import load_frame, open_dataset
from multimodal_perception.geometry import depth_cloud, thermal_correspondence
from multimodal_perception.thermal_fusion import load_fusion_calibration


def validate(root, profile, observations_path, output, report_path):
    cv2.setNumThreads(4)
    dataset = open_dataset(root)
    calibration = load_fusion_calibration(root, profile)
    raw = yaml.safe_load((Path(root) / "calibration/thermal_left.yaml").read_text())
    raw_K = np.array(raw["camera_matrix"]["data"]).reshape(3, 3)
    raw_camera = pinhole_camera(raw_K, 640, 512, calibration["thermal"]["frame"])
    observations = json.loads(Path(observations_path).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in [0, 25, 100, 150, 200, 250, 290]:
        sample = load_frame(dataset, index)
        if not sample["valid"]:
            raise ValueError(f"Validation frame {index} is not valid")
        points, pixels = depth_cloud(sample["depth"], calibration["rgb"], stride=1)
        views = []
        record = {"index": index, "rgb_thermal_skew_ms": abs(sample["rgb_timestamp_s"]-sample["thermal_timestamp_s"])*1000}
        for name, camera in [("source_derived", calibration["thermal"]), ("raw_P_comparison_only", raw_camera)]:
            selected, uv = thermal_correspondence(points, calibration["transform"], camera, calibration["rgb"]["frame"], verified=True)
            warp = np.zeros((512, 640, 3), dtype=np.uint8)
            warp[uv[:, 1], uv[:, 0]] = sample["rgb"][pixels[selected, 1], pixels[selected, 0]]
            record[name + "_dense_coverage"] = len(selected) / (640*512)
            views.append(warp)
        thermal = cv2.cvtColor(sample["thermal"], cv2.COLOR_GRAY2BGR)
        edges_rgb = cv2.Canny(cv2.cvtColor(views[0], cv2.COLOR_BGR2GRAY), 60, 140)
        edges_thermal = cv2.Canny(sample["thermal"], 60, 140)
        overlay = (thermal.astype(float)*.4).astype(np.uint8)
        overlay[edges_rgb > 0] = (0, 255, 0)
        overlay[edges_thermal > 0] = (0, 0, 255)
        for observation in observations["observations"]:
            if observation["index"] != index:
                continue
            y = observation["row_y"]
            for panel, key in [(thermal, "thermal_x"), (views[0], "projected_rgb_x")]:
                x = observation[key]
                cv2.drawMarker(panel, (x, y), (255, 200, 30), cv2.MARKER_CROSS, 19, 1)
                cv2.putText(panel, str(x), (x+5, y+23), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 200, 30), 1, cv2.LINE_AA)
        canvas = np.zeros((1104, 1280, 3), dtype=np.uint8)
        titles = ["Released thermal (marks: manual horizontal observations)", "Source-derived K: projected RGB / CAD nominal pose",
                  "RGB edges green / thermal edges red (spectral edges differ)", "Raw YAML P: comparison only, not deployed"]
        for j, (panel, title) in enumerate(zip([thermal, views[0], overlay, views[1]], titles)):
            x, y = (j % 2)*640, (j // 2)*552
            canvas[y+40:y+552, x:x+640] = panel
            cv2.putText(canvas, title, (x+10, y+25), cv2.FONT_HERSHEY_SIMPLEX, .44, (240, 240, 240), 1, cv2.LINE_AA)
        filename = output / f"alignment_{index:04d}.jpg"
        cv2.imwrite(str(filename), canvas)
        record.update(artifact=str(filename), artifact_sha256=hashlib.sha256(filename.read_bytes()).hexdigest())
        frames.append(record)
    offsets = [abs(o["projected_rgb_x"]-o["thermal_x"]) for o in observations["observations"]]
    report = {"calibration": calibration["evidence"], "frames": frames, "manual_observations": observations,
              "manual_horizontal_abs_error_px": offsets,
              "manual_horizontal_abs_error_mean_px": float(np.mean(offsets)),
              "manual_horizontal_abs_error_max_px": max(offsets),
              "projection_domain": "source reconstructed, hash guarded",
              "physical_registration": "approximate; residual error remains; no fitted extrinsic correction",
              "registration_accuracy_certified": False,
              "interpretation": "Source-derived rectification resolves the image-domain blocker. CAD extrinsics, depth estimates, blur and timing still limit physical alignment. Sparse manual observations are not independent metric ground truth."}
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["projection_domain", "physical_registration", "manual_horizontal_abs_error_px"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce source-derived thermal projection and real-image alignment diagnostics")
    parser.add_argument("--root", default="data/tartan")
    parser.add_argument("--profile", default="configs/tartan_thermal_profile.json")
    parser.add_argument("--observations", default="configs/tartan_alignment_observations.json")
    parser.add_argument("--output", default="results/jetson/tartan/alignment")
    parser.add_argument("--report", default="reports/jetson/tartan/calibration_validation.json")
    args = parser.parse_args()
    validate(args.root, args.profile, args.observations, args.output, args.report)
