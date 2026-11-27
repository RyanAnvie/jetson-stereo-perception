"""Check profile output equivalence and decode every delivered video frame."""
import hashlib
import json
from pathlib import Path
import subprocess

import cv2


def verify():
    root = Path("reports/jetson/tartan_latency")
    summary = json.loads((root/"benchmark_summary.json").read_text())
    baseline = json.loads(Path("reports/jetson/tartan_fixed/fixed_corrected.outputs.json").read_text())
    for row in summary["runs"]:
        report_path = Path(row["report"])
        assert json.loads(report_path.with_suffix(".outputs.json").read_text()) == baseline
        report = json.loads(report_path.read_text())
        assert report["model"]["precision"] == "fp32_no_tf32"
        assert report["accepted_residual_alignment_groups"] == 0
        assert report["dropped_valid_frames"] == report["held_video_frames"] == 0
    results = {}
    for name in ["balanced", "responsive"]:
        video = Path("results/jetson/tartan_latency")/f"{name}.mp4"
        decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"], capture_output=True, text=True)
        (root/f"{name}_decode.log").write_text(decoded.stderr)
        assert decoded.returncode == 0 and not decoded.stderr
        cap = cv2.VideoCapture(str(video))
        count, previous, repeated = 0, None, []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            digest = hashlib.sha256(frame[112:398, 24:624].tobytes()).hexdigest()
            if digest == previous:
                repeated.append(count)
            previous = digest
            if count in [148, 225]:
                cv2.imwrite(str(video.parent/f"{name}_verified_{count:04d}.jpg"), frame)
            count += 1
        cap.release()
        assert count == 296 and not repeated
        results[name] = {"video": str(video), "frames": count, "bytes": video.stat().st_size,
                         "sha256": hashlib.sha256(video.read_bytes()).hexdigest(), "decode_errors": "",
                         "consecutive_identical_rgb_panels": repeated}
    result = {"all_repeated_outputs_exact": True, "repeated_valid_frames": summary["total_valid_frames"], "videos": results}
    (root/"deployment_validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    verify()
