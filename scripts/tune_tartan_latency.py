"""Compare bounded queue depths with unchanged model, geometry and source order."""
import argparse
import json
from pathlib import Path

import yaml

from multimodal_perception.joint_pipeline import run_joint


def tune(cpu_sweep=False, advanced=False):
    root = Path("reports/jetson/tartan_latency")/("advanced_tuning" if advanced else "cpu_tuning" if cpu_sweep else "tuning")
    root.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(Path("reports/jetson/tartan_fixed/fixed_corrected.outputs.json").read_text())
    base = yaml.safe_load(Path("configs/tartan_joint_max_fp32.yaml").read_text())
    variants = [("previous", 4, 3, 2, 2, "inline_refill"),
                ("dispatch_only", 4, 3, 2, 2, "independent"),
                ("dual_a3", 3, 3, 2, 2, "independent"),
                ("dual_a2", 2, 2, 2, 2, "independent"),
                ("dual_a1", 1, 1, 2, 2, "independent"),
                ("single_a2", 2, 2, 1, 2, "independent"),
                ("single_a1", 1, 1, 1, 2, "independent"),
                ("dual_a2_render1", 2, 2, 2, 1, "independent")]
    if cpu_sweep:
        variants = [("dual_inline_a2_cv4", 2, 2, 2, 1, "inline_refill"),
                    ("dual_a2_cv2", 2, 2, 2, 1, "independent"),
                    ("dual_a2_cv6", 2, 2, 2, 1, "independent"),
                    ("dual_a2_cv8", 2, 2, 2, 1, "independent"),
                    ("single_a1_cv6", 1, 1, 1, 1, "independent"),
                    ("single_a1_cv8", 1, 1, 1, 1, "independent")]
    if advanced:
        variants = [("dual_a2_cv8", 2, 2, 2, 1, "independent"),
                    ("dual_a2_cv10", 2, 2, 2, 1, "independent"),
                    ("dual_a2_cv12", 2, 2, 2, 1, "independent"),
                    ("single_a1_cv6", 1, 1, 1, 1, "independent"),
                    ("single_a1_cv8", 1, 1, 1, 1, "independent"),
                    ("single_a1_cv12", 1, 1, 1, 1, "independent")]
    rows = []
    for name, ahead, preparation, inference, render_bound, dispatch in variants:
        config = {**base, "prefetch_frames": ahead, "preparation_workers": preparation,
                  "inference_workers": inference, "inference_dispatch": dispatch, "render_queue_frames": render_bound}
        if cpu_sweep or advanced:
            config["opencv_threads"] = int(name.rsplit("cv", 1)[1])
        path = root/f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        report_path = root/f"{name}.json"
        video = Path("results/jetson/tartan_latency")/root.name/f"{name}.mp4"
        result = run_joint(path, 0, video, report_path)
        output = json.loads(report_path.with_suffix(".outputs.json").read_text())
        assert output == baseline, name
        keys = ["input_to_output_ms", "input_to_encoder_ms", "input_to_detection_ms", "admission_to_encoder_ms",
                "preparation_queue_ms", "preparation_active_ms", "prepared_wait_ms", "inference_active_ms",
                "completed_wait_ms", "render_queue_wait_ms", "render_ms", "encoder_submit_ms"]
        rows.append({"name": name, "fps": result["pipeline_fps"], "stages": {k: result["stages"][k] for k in keys},
                     "output_exact": True, "config": str(path), "report": str(report_path)})
        (root/"summary.json").write_text(json.dumps(rows, indent=2)+"\n")
        print("LATENCY", json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-sweep", action="store_true")
    parser.add_argument("--advanced", action="store_true")
    args = parser.parse_args()
    tune(args.cpu_sweep, args.advanced)
