"""Verify full fixed-rig run/video invariants without rerunning inference."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def verify():
    folder = Path("reports/jetson/tartan_fixed")
    headless = json.loads((folder/"fixed_corrected.json").read_text())
    render = json.loads((folder/"fixed_video.json").read_text())
    first = json.loads((folder/"fixed_corrected.outputs.json").read_text())
    second = json.loads((folder/"fixed_video.outputs.json").read_text())
    calibration = json.loads(Path("configs/tartan_fixed_refinement.json").read_text())
    for report in [headless, render]:
        assert not report["config"]["residual_alignment"] and report["config"]["temporal_compensation"]
        assert report["accepted_residual_alignment_groups"] == 0 and report["selected_valid_groups"] == 296
        np.testing.assert_array_equal(report["calibration"]["rgb_to_thermal_R"], calibration["rgb_to_thermal_R"])
        np.testing.assert_array_equal(report["calibration"]["rgb_to_thermal_t_m"], calibration["rgb_to_thermal_t_m"])
    assert [r["index"] for r in second] == list(range(296))
    assert all(a["detections"] == b["detections"] and a["time_alignment"] == b["time_alignment"] for a, b in zip(first, second))
    assert all(not r["residual_alignment"]["accepted"] and r["residual_alignment"]["delta_uv"] == [0., 0.]
               and r["time_alignment"]["fixed_extrinsics"] for r in second)
    assert render["held_video_frames"] == 0 and render["source_gap_ticks_removed"] == 64
    video = Path("results/jetson/tartan_fixed/tartan_fixed_clock_joint.mp4")
    cap = cv2.VideoCapture(str(video))
    count, previous, repeats = 0, None, []
    while True:
        ok, frame = cap.read()
        if not ok: break
        if count in [0, 15+148, 15+225, 325]:
            cv2.imwrite(str(video.parent/f"verified_frame_{count:04d}.jpg"), frame)
        if 15 <= count < 311:
            digest = hashlib.sha256(frame[112:398, 24:624].tobytes()).hexdigest()
            if digest == previous: repeats.append(count)
            previous = digest
        count += 1
    cap.release()
    metadata = json.loads((folder/"video_metadata.json").read_text())
    assert count == 326 and not repeats and metadata["under_10MB"]
    assert (folder/"video_decode.log").read_text() == ""
    result = {"decoded_frames": count, "valid_content_frames": 296, "title_card_frames": 30,
              "consecutive_identical_valid_rgb_panels": repeats, "held_video_frames": 0, "source_gap_ticks_removed": 64,
              "per_frame_ecc_accepted": 0, "fixed_extrinsics_match_calibration": True,
              "video_matches_headless_detections_and_time_alignment": True,
              "motion_compensated_groups": render["motion_compensated_groups"],
              "uncompensated_motion_groups": render["uncompensated_motion_groups"],
              "duration_s": float(metadata["format"]["duration"]), "bytes": video.stat().st_size,
              "sha256": hashlib.sha256(video.read_bytes()).hexdigest(), "ffmpeg_full_decode_errors": ""}
    (folder/"video_validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    verify()
