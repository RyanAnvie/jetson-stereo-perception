import os
from pathlib import Path

import numpy as np
import pytest

from multimodal_perception.detection import close_detector, decode_output, infer_tensor, load_detector
from multimodal_perception.tensorrt_inference import cuda_check, infer_engine


def test_raw_decoder_preserves_scores_and_box_geometry():
    raw = np.zeros((1, 3549, 85), dtype=np.float32)
    raw[0, 0, :5] = [10, 20, np.log(4), np.log(6), .9]
    raw[0, 0, 7] = .8  # COCO car: objectness * class probability.
    detector = {"grid": np.zeros((3549, 2)), "strides": np.ones((3549, 1))}
    detections = decode_output(raw, 1, (50, 50, 3), detector)
    assert len(detections) == 1 and detections[0]["class"] == "car"
    assert abs(detections[0]["score"] - .72) < 1e-6
    np.testing.assert_allclose(detections[0]["bbox_xyxy"], [8, 17, 12, 23], atol=1e-6)
    assert decode_output(raw, 1, (50, 50, 3), detector, score_threshold=.73) == []
    raw[0, 2, 0] = np.nan
    with pytest.raises(ValueError, match="Invalid YOLOX output"):
        decode_output(raw, 1, (50, 50, 3), detector)


def test_backend_configuration_rejects_silent_fallback(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"placeholder")
    with pytest.raises(ValueError, match="Unknown detector backend"):
        load_detector(model, backend="invalid")
    with pytest.raises(ValueError, match="requires --engine"):
        load_detector(model, backend="tensorrt")
    with pytest.raises(ValueError, match="requires the tensorrt"):
        load_detector(model, engine="model.engine")


def test_cuda_errors_and_closed_engine_are_explicit():
    assert cuda_check((0, 42)) == 42
    with pytest.raises(RuntimeError, match="CUDA call failed"):
        cuda_check((1,))
    with pytest.raises(RuntimeError, match="closed"):
        infer_engine({"closed": True}, np.zeros((1, 3, 416, 416), np.float32))


def test_target_engines_execute_repeatably_and_release_resources():
    directory = os.environ.get("JETSON_TEST_ENGINE_DIR")
    if not directory:
        pytest.skip("Set JETSON_TEST_ENGINE_DIR on the target to exercise actual FP32 and FP16 engines")
    tensor = np.full((1, 3, 416, 416), 114, np.float32)
    for precision in ["fp32", "fp16"]:
        detector = load_detector("models/weights/yolox_nano.onnx", backend="tensorrt",
                                 engine=Path(directory) / f"yolox_nano_{precision}.engine")
        try:
            first, timing = infer_tensor(tensor, detector)
            second, _ = infer_tensor(tensor, detector)
            assert first.shape == (1, 3549, 85) and np.isfinite(first).all()
            np.testing.assert_array_equal(first, second)
            assert all(value >= 0 for value in timing.values())
        finally:
            close_detector(detector)
        assert detector["trt"]["closed"] and not detector["trt"]["buffers"]
