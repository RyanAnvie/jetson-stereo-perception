"""Verify the final hardware-encoded video and repeated output consistency."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

import cv2
import numpy as np


def verify():
    root = Path("reports/jetson/tartan_fast")
    summary = json.loads((root/"sustained_summary.json").read_text())
    reference = json.loads(Path("reports/jetson/tartan_fixed/fixed_corrected.outputs.json").read_text())
    calibration = json.loads(Path("configs/tartan_fixed_refinement.json").read_text())
    for run in summary["runs"]:
        path = Path(run["report"])
        report = json.loads(path.read_text())
        actual = json.loads(path.with_suffix(".outputs.json").read_text())
        assert actual == reference and [r["index"] for r in actual] == list(range(296))
        assert report["pipeline_fps"] > 15 and report["dropped_valid_frames"] == 0
        assert report["held_video_frames"] == 0 and report["accepted_residual_alignment_groups"] == 0
        assert report["config"]["registration_backend"] == "cuda"
        assert report["model"]["precision"] == "fp32_no_tf32"
        np.testing.assert_array_equal(report["calibration"]["rgb_to_thermal_R"], calibration["rgb_to_thermal_R"])
        np.testing.assert_array_equal(report["calibration"]["rgb_to_thermal_t_m"], calibration["rgb_to_thermal_t_m"])
    video = Path("results/jetson/tartan_fast/tartan_fast_fp32.mp4")
    metadata = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(video)], check=True, capture_output=True, text=True).stdout)
    (root/"video_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"], capture_output=True, text=True)
    (root/"video_decode.log").write_text(decoded.stderr)
    assert decoded.returncode == 0 and decoded.stderr == ""
    cap = cv2.VideoCapture(str(video))
    count, previous, repeats = 0, None, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if count in [0, 148, 225, 295]:
            cv2.imwrite(str(video.parent/f"verified_frame_{count:04d}.jpg"), frame)
        digest = hashlib.sha256(frame[112:398, 24:624].tobytes()).hexdigest()
        if digest == previous:
            repeats.append(count)
        previous = digest
        count += 1
    cap.release()
    assert count == 296 and not repeats and video.stat().st_size < 10_000_000
    # Older tegrastats builds expose these individual rails, not VDD_IN.
    raw = (root/"tegrastats.log").read_text()
    rails = {}
    for rail in ["VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0"]:
        values = [float(x)/1000 for x in re.findall(rail+r" (\d+)mW", raw)]
        if values:
            rails[rail] = {"p50": float(np.median(values)), "max": max(values)}
    result = {"repeated_valid_groups": summary["total_valid_groups"],
              "all_repeated_outputs_bit_exact_to_original_fp32": True, "all_runs_above_15_fps": True,
              "decoded_video_frames": count, "consecutive_identical_rgb_panels": repeats,
              "source_gap_ticks_removed": 64, "held_video_frames": 0, "title_frames": 0,
              "video_duration_s": float(metadata["format"]["duration"]), "video_bytes": video.stat().st_size,
              "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
              "codec": metadata["streams"][0]["codec_name"], "ffmpeg_decode_errors": "",
              "individual_power_rails_w": rails,
              "gpu_clock_after_run_hz": int(Path("/sys/class/devfreq/17000000.gpu/cur_freq").read_text()),
              "cpu0_clock_after_run_khz": int(Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").read_text())}
    (root/"deployment_validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    verify()
