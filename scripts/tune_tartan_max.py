"""Measure complete-sequence CPU/GPU concurrency without changing numerics."""
import argparse
import json
from pathlib import Path

import yaml

from multimodal_perception.joint_pipeline import run_joint


def tune(render=False):
    root = Path("reports/jetson/tartan_max/tuning_render" if render else "reports/jetson/tartan_max/tuning")
    root.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(Path("reports/jetson/tartan_fixed/fixed_corrected.outputs.json").read_text())
    base = yaml.safe_load(Path("configs/tartan_joint_max_fp32.yaml").read_text())
    variants = [("dual_mapped_2x4", 2, 4, 2), ("dual_mapped_3x2", 3, 2, 2),
                ("dual_mapped_3x4", 3, 4, 2), ("dual_mapped_4x2", 4, 2, 2),
                ("triple_mapped_3x2", 3, 2, 3)]
    if render:
        variants = [("dual_mapped_2x4", 2, 4, 2), ("dual_mapped_3x4", 3, 4, 2), ("triple_mapped_2x4", 2, 4, 3)]
    rows = []
    for name, preparation, opencv, inference in variants:
        config = {**base, "cuda_mapped_outputs": True, "preparation_workers": preparation,
                  "opencv_threads": opencv, "inference_workers": inference}
        path = root/f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        report_path = root/f"{name}.json"
        video = Path("results/jetson/tartan_max")/f"tune_{name}.mp4" if render else None
        result = run_joint(path, 0, video, report_path)
        output = json.loads(report_path.with_suffix(".outputs.json").read_text())
        changed = [a["index"] for a, b in zip(output, baseline) if a != b]
        assert len(output) == len(baseline) and not changed
        rows.append({"name": name, "fps": result["pipeline_fps"], "changed_frames": changed,
                     "config": str(path), "report": str(report_path), "peak_rss_kib": result["process_peak_rss_kib"]})
        (root/"summary.json").write_text(json.dumps(rows, indent=2)+"\n")
        print("TUNING", json.dumps(rows[-1]), flush=True)
    print(json.dumps(sorted(rows, key=lambda r: -r["fps"]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    tune(parser.parse_args().render)
