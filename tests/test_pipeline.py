import json

import cv2
import numpy as np
import pytest

from multimodal_perception.camera import load_calibration
from multimodal_perception.dataset import open_dataset, load_frame
from multimodal_perception.replay import summarize_timings


def test_percentiles_are_from_frame_samples():
    rows = [{"index": i, "processing_ms": value} for i, value in enumerate([1., 2., 3., 4., 100.])]
    result = summarize_timings(rows, 2.5)
    assert result["pipeline_fps"] == 2
    assert result["stages"]["processing_ms"]["p50"] == 3
    assert np.isclose(result["stages"]["processing_ms"]["p95"], 80.8)
    with pytest.raises(ValueError):
        summarize_timings([], 1)


def test_dataset_ffc_missing_file_and_depth_contract(tmp_path):
    sequence = tmp_path / "fixture"
    sequence.mkdir()
    (tmp_path / "sample_manifest.json").write_text(json.dumps({"sequence": "fixture", "revision": "synthetic"}))
    np.savetxt(sequence / "target_timestamps.txt", [100., 100.1, 100.2])
    for name, frame in [("zed_left_rect", "zed_left_camera_optical_frame"), ("thermal_left_rect_8", "thermal_left/optical_frame")]:
        folder = sequence / name
        folder.mkdir()
        (folder / "frames.yaml").write_text("frame_id: " + frame)
        np.savetxt(folder / "timestamps.txt", [100.01, 100.11, 100.21])
        np.savetxt(folder / "errors.txt", [0.01, 0.01, 0.01])
        shape = (540, 960, 3) if name == "zed_left_rect" else (512, 640)
        for index in range(3):
            cv2.imwrite(str(folder / f"{index:08d}.png"), np.zeros(shape, dtype=np.uint8))
    (sequence / "thermal_left_ffc").mkdir()
    np.savetxt(sequence / "thermal_left_ffc/data.txt", [0, 1, 0])
    np.savetxt(sequence / "thermal_left_ffc/errors.txt", [0.01, 0.01, 0.01])
    (sequence / "stereo_depth").mkdir()
    for index in [0, 1]:
        np.save(sequence / "stereo_depth" / f"{index:08d}.npy", np.full((540, 960), 2, dtype=np.float32))
    dataset = open_dataset(tmp_path)
    assert dataset["valid"].tolist() == [True, False, False]
    assert dataset["reasons"][1] == ["ffc_or_ffc_timestamp_invalid"]
    assert dataset["reasons"][2] == ["stereo_depth_missing"]
    sample = load_frame(dataset, 0)
    assert sample["depth"].dtype == np.float32 and sample["units"] == "m"
    np.save(sequence / "stereo_depth/00000000.npy", np.ones((540, 960), dtype=np.float64))
    with pytest.raises(ValueError, match="dtype"):
        load_frame(dataset, 0)


def test_published_tf_pose_is_inverted(tmp_path):
    (tmp_path / "released_zed_rectified_960x540.yaml").write_text("released_resolution: [960, 540]\nrgb_left_rectified:\n  K: [369.431, 0, 484.2525, 0, 369.4055, 295.99275, 0, 0, 1]\n")
    tf = {"transforms": [{"parent": "zed_left_camera_optical_frame", "child": "thermal_left/optical_frame", "translation_xyz_m": [0, 0.03166579, 0], "quaternion_xyzw": [0, 0, 0, 1]}]}
    (tmp_path / "tf_static.json").write_text(json.dumps(tf))
    camera, transform = load_calibration(tmp_path)
    assert transform["source"] == camera["frame"]
    np.testing.assert_allclose(transform["t"], [0, -0.03166579, 0])
