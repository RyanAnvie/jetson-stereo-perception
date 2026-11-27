import os
import cv2
import numpy as np
import pytest

from multimodal_perception.joint_detection import (
    preprocess_joint, decode_joint, attach_joint_geometry, load_joint_detector, infer_joint, close_joint_detector, process_joint_sample)
from multimodal_perception.residual_alignment import estimate_residual_shift, apply_residual_shift
from multimodal_perception.tensorrt_inference import infer_engine_multi


def test_two_modalities_preprocessing_and_ablation_are_independent():
    rgb = np.zeros((512, 640, 3), np.uint8)
    rgb[:, :, 2] = 255
    heat = np.tile(np.arange(640, dtype=np.uint16) % 256, (512, 1)).astype(np.uint8)
    both = preprocess_joint(rgb, heat)
    assert set(both) == {"rgb", "thermal"}
    np.testing.assert_array_equal(both["rgb"][0, :, 64, 0], [1, 0, 0])
    np.testing.assert_allclose(both["thermal"][0, :, 64, 5], [5/255]*3)
    assert both["rgb"].shape == (1, 3, 640, 640)
    blank = preprocess_joint(rgb, heat, "thermal_flat")
    np.testing.assert_array_equal(blank["rgb"], both["rgb"])
    assert np.std(blank["thermal"][0, :, 64:576]) < 1e-6
    assert np.std(both["thermal"][0, :, 64:576]) > .1


def test_joint_decoder_class_order_letterbox_and_nms():
    classes = ["People", "Car", "Bus", "Lamp", "Motorcycle", "Truck"]
    raw = np.zeros((1, 102000, 11), np.float32)
    raw[0, 0, :5] = [320, 320, 100, 100, .9]
    raw[0, 0, 8] = .8
    raw[0, 1] = raw[0, 0]
    raw[0, 1, 4] = .7
    detections = decode_joint(raw, classes)
    assert len(detections) == 1 and detections[0]["class"] == "Lamp"
    np.testing.assert_array_equal(detections[0]["bbox_xyxy"], [270, 206, 370, 306])
    assert detections[0]["recognition_inputs"] == ["rgb", "thermal"]
    assert abs(detections[0]["score"]-.72) < 1e-6
    raw[0, 2, 0] = np.nan
    with pytest.raises(ValueError):
        decode_joint(raw, classes)


def test_joint_geometry_uses_thermal_boxes_and_visible_points():
    detections = [{"bbox_xyxy": [10, 10, 20, 20]}, {"bbox_xyxy": [100, 100, 120, 120]}]
    fusion = {"thermal_pixels": np.array([[12, 13], [14, 15], [150, 150]]),
              "rgb_pixels": np.array([[25, 26], [27, 28], [200, 200]]),
              "points": np.array([[0, 0, 2], [2, 2, 4], [50, 50, 100]]),
              "intensity_u8": np.array([80, 100, 255], np.uint8)}
    attach_joint_geometry(detections, fusion, {"rgb": {"frame": "rgb"}}, min_points=2)
    assert detections[0]["position"]["point_m"] == [1, 1, 3]
    assert detections[0]["rgb_bbox_xyxy"] == [25, 26, 28, 29]
    assert detections[1]["position"]["status"] == "insufficient_depth"


def test_ecc_recovers_polarity_inverted_translation_and_rejects_blank():
    image = np.zeros((512, 640), np.uint8)
    rng = np.random.default_rng(7)
    for _ in range(35):
        a = tuple(rng.integers([50, 50], [540, 412]).tolist())
        b = (a[0]+int(rng.integers(15, 65)), a[1]+int(rng.integers(15, 65)))
        cv2.rectangle(image, a, b, int(rng.integers(80, 255)), 3)
    thermal = 255-cv2.warpAffine(image, np.float32([[1, 0, 7], [0, 1, -4]]), (640, 512))
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    observed = np.full(image.shape, 255, np.uint8)
    result = estimate_residual_shift(rgb, thermal, observed)
    assert result["accepted"]
    np.testing.assert_allclose(result["delta_uv"], [7, -4], atol=.5)
    blank = estimate_residual_shift(np.zeros_like(rgb), np.zeros_like(image), observed)
    assert not blank["accepted"]


def test_residual_shift_updates_geometry_and_thermal_samples_consistently():
    heat = np.tile(np.arange(640, dtype=np.uint16) % 256, (512, 1)).astype(np.uint8)
    fusion = {"thermal_pixels": np.array([[10, 10], [639, 511]]), "rgb_pixels": np.array([[20, 20], [30, 30]]),
              "points": np.array([[0, 0, 2], [1, 1, 3]]), "intensity_u8": np.array([10, 127], np.uint8)}
    rgb = np.zeros((512, 640, 3), np.uint8)
    mask = np.ones((512, 640), np.uint8)*255
    alignment = {"accepted": True, "delta_uv": [7., -4.]}
    apply_residual_shift({"thermal": heat}, fusion, rgb, mask, mask, alignment)
    assert fusion["thermal_pixels"].tolist() == [[17, 6]]
    assert fusion["intensity_u8"].tolist() == [17]
    assert fusion["points"].tolist() == [[0, 0, 2]]


def test_multi_input_adapter_rejects_missing_modality():
    with pytest.raises(ValueError, match="names"):
        infer_engine_multi({"input_names": ["rgb", "thermal"]}, {"rgb": np.zeros(1)})


def test_fixed_clock_mode_forbids_ecc_and_excludes_unbounded_depth_positions():
    with pytest.raises(ValueError, match="must not use per-frame ECC"):
        process_joint_sample({"valid": True, "ffc": False}, None, None,
                             {"temporal_compensation": True, "residual_alignment": True})
    detections = [{"bbox_xyxy": [0, 0, 20, 20]}]
    fusion = {"thermal_pixels": np.array([[5, 5], [6, 6], [7, 7]]),
              "rgb_pixels": np.array([[10, 10], [11, 11], [12, 12]]),
              "points": np.array([[0, 0, 4], [1, 1, 6], [100, 100, 100000]]),
              "intensity_u8": np.array([10, 20, 30], np.uint8)}
    attach_joint_geometry(detections, fusion, {"rgb": {"frame": "rgb"}}, 2, max_depth_m=30.)
    assert detections[0]["position"]["point_m"] == [.5, .5, 5.]
    assert detections[0]["position"]["range_excluded_depth_samples"] == 1


def test_real_joint_engine_uses_both_inputs_and_releases_resources():
    if not os.environ.get("JETSON_TEST_JOINT_MODEL"):
        pytest.skip("Set JETSON_TEST_JOINT_MODEL on the target")
    detector = load_joint_detector("models/weights/lcafnet_m3fd.onnx", "models/weights/jetson/lcafnet_m3fd_fp32.engine")
    rng = np.random.default_rng(5)
    tensors = {name: rng.random((1, 3, 640, 640), dtype=np.float32) for name in ["rgb", "thermal"]}
    try:
        original, _ = infer_joint(detector, tensors)
        repeat, _ = infer_joint(detector, tensors)
        np.testing.assert_array_equal(original, repeat)
        for modality in ["rgb", "thermal"]:
            changed = {name: value.copy() for name, value in tensors.items()}
            changed[modality].fill(.5)
            ablation, _ = infer_joint(detector, changed)
            assert np.max(np.abs(ablation[:, :, 4:]-original[:, :, 4:])) > 1e-4
    finally:
        close_joint_detector(detector)
    assert detector["trt"]["closed"] and not detector["trt"]["buffers"]
