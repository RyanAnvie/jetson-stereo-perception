import os
import time

import cv2
import numpy as np
import pytest

from multimodal_perception.camera import pinhole_camera, rigid_transform


@pytest.mark.parametrize("dispatch,worker_count", [("inline_refill", 2), ("independent", 2), ("independent", 1)])
def test_parallel_gpu_dispatch_keeps_order_owns_states_and_closes_source(monkeypatch, dispatch, worker_count):
    import multimodal_perception.joint_workers as module
    monkeypatch.setattr(module, "activate_cuda", lambda: None)
    active, closed = set(), []
    def process(sample, detector, calibration, config):
        identity = detector["identity"]
        assert identity not in active
        active.add(identity)
        try:
            time.sleep(.005 if sample["index"] == 0 else .001)
            return [{"source": sample["index"]}], {"owner": identity}, {"processing_ms": 1.}
        finally:
            active.remove(identity)
    monkeypatch.setattr(module, "process_joint_sample", process)
    def source():
        try:
            yield from ({"index": i} for i in range(8))
        finally:
            closed.append(True)
    detectors = [{"backend": "tensorrt", "identity": i} for i in range(worker_count)]
    result = list(module.processed_samples(source(), detectors, {}, {"inference_dispatch": dispatch}))
    assert [r[0]["index"] for r in result] == list(range(8))
    assert [r[2]["owner"] for r in result] == [i % worker_count for i in range(8)]
    assert closed == [True] and not active
    stream = module.processed_samples(source(), detectors, {}, {"inference_dispatch": dispatch})
    next(stream)
    stream.close()
    assert len(closed) == 2 and not active


@pytest.mark.parametrize("mapped", [False, True])
def test_gpu_compaction_all_hole_ties_and_masks_match_original(mapped):
    if not os.environ.get("JETSON_TEST_CUDA_REGISTRATION"):
        pytest.skip("Requires target CUDA")
    from multimodal_perception.cuda_registration import CudaRegistration
    from multimodal_perception.thermal_fusion import fuse_thermal
    from multimodal_perception.joint_detection import registered_rgb
    h, w = 512, 640
    depth = np.full((h, w), np.nan, np.float32)
    offsets = [(0, -1), (-1, 0), (0, 1), (1, 0), (-1, -1), (1, -1), (1, 1), (-1, 1), (0, -2), (-2, 0), (0, 2), (2, 0)]
    for bits in range(4096):
        y, x = (bits//80)*8+3, (bits%80)*8+3
        for j, (dx, dy) in enumerate(offsets):
            if bits & (1 << j):
                depth[y+dy, x+dx] = 1
    # Include boundary holes, zero/negative/infinite depths and exact image edges.
    depth[-1, -1], depth[-2, 0], depth[0, -1] = 1, 1, 1
    depth[-1, :3] = [0, -1, np.inf]
    ids = np.arange(h*w).reshape(h, w)
    rgb = np.stack((ids % 256, (ids//256) % 256, ids//65536), axis=2).astype(np.uint8)
    sample = {"timestamp_s": 1., "rgb_timestamp_s": 1., "thermal_timestamp_s": 1., "valid": True, "ffc": False,
              "rgb": rgb, "thermal": (ids % 256).astype(np.uint8), "depth": depth}
    calibration = {"rgb": pinhole_camera(np.eye(3), w, h, "rgb"), "thermal": pinhole_camera(np.eye(3), w, h, "thermal"),
                   "transform": rigid_transform(np.eye(3), [0, 0, 0], "rgb", "thermal")}
    expected = fuse_thermal(sample, [], calibration, stride=1)
    image, observed, usable = registered_rgb(sample, expected)
    gpu = CudaRegistration((h, w), (h, w), compact=True, mapped_outputs=mapped)
    try:
        actual, new_image, new_observed = gpu.fuse(sample, calibration)
        for key in ["points", "rgb_pixels", "thermal_pixels", "intensity_u8"]:
            np.testing.assert_array_equal(actual[key], expected[key])
        np.testing.assert_array_equal(new_image, image)
        np.testing.assert_array_equal(new_observed, observed)
        np.testing.assert_array_equal(actual["gpu_usable_mask"], usable)
        old = new_image.copy()
        sample["depth"][:] = np.nan
        empty, blank, mask = gpu.fuse(sample, calibration)
        assert len(empty["points"]) == 0 and not mask.any() and (blank == 114).all()
        np.testing.assert_array_equal(old, new_image)
    finally:
        gpu.close()


@pytest.mark.parametrize("mapped", [False, True])
def test_gpu_candidate_filter_preserves_all_candidates_and_checks_hidden_nan(mapped):
    if not os.environ.get("JETSON_TEST_CUDA_REGISTRATION"):
        pytest.skip("Requires target CUDA")
    from multimodal_perception.cuda_registration import CudaRegistration
    from multimodal_perception.tensorrt_inference import cuda_check
    gpu = CudaRegistration(compact=True, mapped_outputs=mapped)
    gpu._allocate("test_predictions", (1, 102000, 11), np.float32, True)
    state = {"stream": gpu.stream, "buffers": {"predictions": gpu.buffers["test_predictions"]}}
    raw = np.random.default_rng(12).random((1, 102000, 11), dtype=np.float32)
    def run(value):
        b = state["buffers"]["predictions"]
        np.copyto(b["host"], value)
        cuda_check(gpu.cuda.cudaMemcpyAsync(b["device"], b["host_pointer"], b["nbytes"], gpu.cuda.cudaMemcpyKind.cudaMemcpyHostToDevice, gpu.stream))
        read = gpu.filter_predictions(state, .25)
        cuda_check(gpu.cuda.cudaStreamSynchronize(gpu.stream))
        return read()["predictions"]
    try:
        selected = (raw[0, :, 4:5]*raw[0, :, 5:]).max(axis=1) >= .25
        assert selected.sum() > 4096
        np.testing.assert_array_equal(run(raw), raw[:, selected])
        raw.fill(1)
        np.testing.assert_array_equal(run(raw), raw)  # No capacity/top-K truncation.
        raw.fill(0)
        assert run(raw).shape == (1, 0, 11)
        raw[0, 45000, 2] = np.nan
        with pytest.raises(ValueError, match="Invalid joint"):
            run(raw)
    finally:
        gpu.close()


def test_cuda_graph_replay_is_exact_and_released():
    if not os.environ.get("JETSON_TEST_JOINT_MODEL"):
        pytest.skip("Requires target TensorRT model")
    from multimodal_perception.joint_detection import load_joint_detector, close_joint_detector, infer_joint
    detector = load_joint_detector("models/weights/lcafnet_m3fd.onnx", "models/weights/jetson/lcafnet_m3fd_fp32.engine")
    rng = np.random.default_rng(7)
    tensors = {k: rng.random((1, 3, 640, 640), dtype=np.float32) for k in ["rgb", "thermal"]}
    try:
        reference, _ = infer_joint(detector, tensors)
        detector["trt"]["cuda_graph_requested"] = True
        first, _ = infer_joint(detector, tensors)
        second, _ = infer_joint(detector, tensors)
        assert detector["trt"]["graph_instance"] is not None
        np.testing.assert_array_equal(reference, first)
        np.testing.assert_array_equal(reference, second)
    finally:
        close_joint_detector(detector)
    assert detector["trt"].get("graph_instance") is None and detector["trt"].get("graph") is None
