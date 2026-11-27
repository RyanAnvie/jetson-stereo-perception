"""Verify sustained exact FP32 output, fixed calibration and the delivered video."""
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


def verify():
    root = Path("reports/jetson/tartan_max")
    summary = json.loads((root/"sustained_summary.json").read_text())
    reference = json.loads(Path("reports/jetson/tartan_fixed/fixed_corrected.outputs.json").read_text())
    calibration_path = Path("configs/tartan_fixed_refinement.json")
    calibration = json.loads(calibration_path.read_text())
    assert hashlib.sha256(calibration_path.read_bytes()).hexdigest() == "513a9ceb81f4f71873f242d217f44028b0be24b9e471fb7a0a930afd7d705dbc"
    assert summary["timed_processing_seconds"] >= 300
    assert summary["persistent_contexts_reused"] == 2
    registration = json.loads((root/"registration_equivalence.json").read_text())
    assert registration["frames"] == 296 and registration["all_images_points_and_masks_exact"]
    for run in summary["runs"]:
        path = Path(run["report"])
        report = json.loads(path.read_text())
        actual = json.loads(path.with_suffix(".outputs.json").read_text())
        assert actual == reference and [r["index"] for r in actual] == list(range(296))
        assert report["pipeline_fps"] > 15 and report["dropped_valid_frames"] == 0
        assert report["held_video_frames"] == 0 and report["accepted_residual_alignment_groups"] == 0
        assert report["config"]["registration_backend"] == "cuda"
        assert report["config"]["cuda_mapped_outputs"] and report["inference_workers"] == 2
        assert report["model"]["precision"] == "fp32_no_tf32" and report["model"]["cuda_graph"]
        np.testing.assert_array_equal(report["calibration"]["rgb_to_thermal_R"], calibration["rgb_to_thermal_R"])
        np.testing.assert_array_equal(report["calibration"]["rgb_to_thermal_t_m"], calibration["rgb_to_thermal_t_m"])
    video = Path("results/jetson/tartan_max/tartan_max_fp32.mp4")
    metadata = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(video)], check=True, capture_output=True, text=True).stdout)
    stream = next(s for s in metadata["streams"] if s["codec_type"] == "video")
    assert stream["codec_name"] == "h264" and (stream["width"], stream["height"]) == (1280, 800)
    assert int(stream["nb_frames"]) == 296
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
    assert count == 296 and not repeats
    result = {"repeated_valid_groups": summary["total_valid_groups"],
              "timed_processing_seconds": summary["timed_processing_seconds"],
              "all_repeated_outputs_bit_exact_to_original_fp32": True, "all_runs_above_15_fps": True,
              "decoded_video_frames": count, "consecutive_identical_rgb_panels": repeats,
              "source_gap_ticks_removed": report["source_gap_ticks_removed"], "held_video_frames": 0, "title_frames": 0,
              "video_duration_s": float(metadata["format"]["duration"]), "video_bytes": video.stat().st_size,
              "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
              "codec": stream["codec_name"], "ffmpeg_decode_errors": "",
              "observed_gpu_clocks_hz": summary["observed_gpu_clocks_hz"],
              "cpu0_clock_after_run_khz": int(Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").read_text())}
    (root/"deployment_validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    verify()
