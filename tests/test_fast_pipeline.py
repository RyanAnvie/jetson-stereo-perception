import os

import cv2
import numpy as np
import pytest

from multimodal_perception.camera import pinhole_camera, rigid_transform
from multimodal_perception.joint_prefetch import prepared_samples
from multimodal_perception.thermal_fusion import fuse_thermal


def test_vectorized_cloud_display_preserves_clipping_painter_order_and_ties():
    from multimodal_perception.dashboard import draw_cloud_pixels
    rng = np.random.default_rng(4)
    uv = rng.integers(-3, 43, (1500, 2))
    colors = rng.integers(0, 256, (1500, 3), dtype=np.uint8)
    order = np.argsort(-rng.integers(1, 5, 1500), kind="stable")
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < 40) & (uv[:, 1] >= 0) & (uv[:, 1] < 40)
    expected = np.zeros((40, 40, 3), np.uint8)
    for i in order:
        if inside[i]:
            cv2.circle(expected, tuple(uv[i]), 1, tuple(int(c) for c in colors[i]), -1)
    actual = np.zeros_like(expected)
    draw_cloud_pixels(actual, uv, colors, order, inside)
    np.testing.assert_array_equal(actual, expected)


def test_prefetch_preserves_order_reuses_neighbor_and_propagates_failure(monkeypatch):
    import multimodal_perception.joint_prefetch as module
    loaded = []
    def load(dataset, index):
        loaded.append(index)
        return {"index": index, "rgb_timestamp_s": index*.1, "thermal_timestamp_s": index*.1}
    def estimate(sample, calibration, enabled):
        assert sample["rgb_motion_neighbor"]["index"] == sample["index"]+1
        return calibration, {"compensated": True, "index": sample["index"]}
    monkeypatch.setattr(module, "load_frame", load)
    monkeypatch.setattr(module, "calibration_at_exposure", estimate)
    dataset = {"target": np.arange(11)*.1, "zed_left_rect": {"actual": np.arange(11)*.1}}
    calibration = {"evidence": {"refinement": {"thermal_clock_correction_s": .07}}}
    samples = list(prepared_samples(dataset, list(range(10)), calibration, True, 2))
    assert [s["index"] for s in samples] == list(range(10))
    assert loaded == list(range(11))  # Adjacent RGB is decoded once, not twice.
    assert [s["prepared_exposure"][1]["index"] for s in samples] == list(range(10))
    with pytest.raises(ValueError, match="1 to 8"):
        list(prepared_samples(dataset, [0], calibration, True, 0))
    def fail(*args):
        raise RuntimeError("decode failure")
    monkeypatch.setattr(module, "load_frame", fail)
    with pytest.raises(RuntimeError, match="decode failure"):
        list(prepared_samples(dataset, [0, 1], calibration, True, 2))


def test_cuda_visibility_ties_invalid_depth_and_buffer_ownership():
    if not os.environ.get("JETSON_TEST_CUDA_REGISTRATION"):
        pytest.skip("CUDA registration requires target NVIDIA runtime")
    from multimodal_perception.cuda_registration import CudaRegistration
    rgb = pinhole_camera(np.eye(3), 10, 10, "rgb")
    thermal = pinhole_camera([[.1, 0, 0], [0, .1, 0], [0, 0, 1]], 10, 10, "thermal")
    calibration = {"rgb": rgb, "thermal": thermal,
                   "transform": rigid_transform(np.eye(3), [0, 0, 0], "rgb", "thermal")}
    depth = np.full((10, 10), np.nan, np.float32)
    depth[0, :6] = [2, 1, 1, 0, -1, np.inf]
    sample = {"timestamp_s": 1., "rgb_timestamp_s": 1., "thermal_timestamp_s": 1.002,
              "valid": True, "ffc": False, "depth": depth,
              "rgb": np.arange(300).reshape(10, 10, 3).astype(np.uint8),
              "thermal": np.full((10, 10), 123, np.uint8)}
    gpu = CudaRegistration((10, 10), (10, 10))
    try:
        expected = fuse_thermal(sample, [], calibration, stride=1)
        actual, image, mask = gpu.fuse(sample, calibration)
        assert actual["rgb_pixels"].tolist() == [[1, 0]]  # nearest Z, then first index
        for key in ["points", "thermal_pixels", "rgb_pixels", "intensity_u8"]:
            np.testing.assert_array_equal(actual[key], expected[key])
        saved = image.copy()
        sample["depth"][:] = np.nan
        empty, blank, empty_mask = gpu.fuse(sample, calibration)
        assert len(empty["points"]) == 0 and not empty_mask.any() and (blank == 114).all()
        np.testing.assert_array_equal(image, saved)
        assert actual["points"][0, 2] == 1 and mask[0, 0] == 255
        for field, value in [("ffc", True), ("valid", False), ("rgb_timestamp_s", np.nan)]:
            with pytest.raises(ValueError, match="Unsynchronized"):
                gpu.fuse({**sample, field: value}, calibration)
    finally:
        gpu.close()
        gpu.close()
    assert gpu.closed and not gpu.buffers


def test_cuda_preprocessing_and_raw_fp32_outputs_are_bit_exact():
    if not os.environ.get("JETSON_TEST_CUDA_REGISTRATION"):
        pytest.skip("CUDA preprocessing requires target TensorRT model")
    from multimodal_perception.cuda_registration import CudaRegistration
    from multimodal_perception.joint_detection import load_joint_detector, close_joint_detector, preprocess_joint, infer_joint
    from multimodal_perception.tensorrt_inference import cuda_check
    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, (512, 640, 3), dtype=np.uint8)
    thermal = rng.integers(0, 256, (512, 640), dtype=np.uint8)
    detector = load_joint_detector("models/weights/lcafnet_m3fd.onnx", "models/weights/jetson/lcafnet_m3fd_fp32.engine")
    gpu = CudaRegistration()
    try:
        tensors = preprocess_joint(rgb, thermal)
        expected, _ = infer_joint(detector, tensors)
        gpu.prepare_engine_inputs(rgb, thermal, detector["trt"])
        state = detector["trt"]
        cuda = state["cuda"]
        for name in ["rgb", "thermal"]:
            b = state["buffers"][name]
            cuda_check(cuda.cudaMemcpyAsync(b["host_pointer"], b["device"], b["host"].nbytes,
                                            cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost, state["stream"]))
        cuda_check(cuda.cudaStreamSynchronize(state["stream"]))
        for name in ["rgb", "thermal"]:
            np.testing.assert_array_equal(state["buffers"][name]["host"], tensors[name])
        actual, _ = infer_joint(detector, None)
        np.testing.assert_array_equal(actual, expected)
    finally:
        gpu.close()
        close_joint_detector(detector)
