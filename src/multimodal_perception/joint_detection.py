import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from multimodal_perception.detection import nonmaximum_suppression
from multimodal_perception.thermal_fusion import fuse_thermal
from multimodal_perception.residual_alignment import estimate_residual_shift, apply_residual_shift
from multimodal_perception.rig_motion import calibration_at_exposure


def load_joint_detector(model_path, engine_path=None, backend="tensorrt", threads=4):
    path = Path(model_path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != metadata["model_sha256"]:
        raise ValueError("Joint model checksum mismatch")
    if metadata["inputs"] != {"rgb": [1, 3, 640, 640], "thermal": [1, 3, 640, 640]}:
        raise ValueError("Unexpected joint input contract")
    if metadata["outputs"] != {"predictions": [1, 102000, 11]} or len(metadata["classes"]) != 6:
        raise ValueError("Unexpected joint output contract")
    state = {"backend": backend, "metadata": metadata}
    if backend == "tensorrt":
        from multimodal_perception.tensorrt_inference import load_engine
        if not engine_path:
            raise ValueError("Joint TensorRT inference requires an engine")
        state["trt"] = load_engine(engine_path, digest, metadata)
        state["metadata"] = {**metadata, **state["trt"]["metadata"]}
    elif backend == "onnxruntime":
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        state["session"] = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        if {item.name: item.shape for item in state["session"].get_inputs()} != metadata["inputs"]:
            raise ValueError("ONNX inputs differ from the model contract")
        state["metadata"].update(backend=backend, runtime=ort.__version__, precision="float32", provider="CPUExecutionProvider")
    else:
        raise ValueError("Unknown joint detector backend")
    return state


def close_joint_detector(detector):
    if "cuda_registration" in detector:
        detector["cuda_registration"].close()
    if detector["backend"] == "tensorrt":
        from multimodal_perception.tensorrt_inference import close_engine
        close_engine(detector["trt"])


def registered_rgb(sample, fusion):
    uv, pixels = fusion["thermal_pixels"], fusion["rgb_pixels"]
    image = np.full((512, 640, 3), 114, dtype=np.uint8)
    observed = np.zeros((512, 640), dtype=np.uint8)
    if not len(uv):
        return image, observed, observed
    image[uv[:, 1], uv[:, 0]] = sample["rgb"][pixels[:, 1], pixels[:, 0]]
    observed[uv[:, 1], uv[:, 0]] = 255
    return fill_registered_holes(image, observed)


def fill_registered_holes(image, observed):
    # Fill at most 2 px raster sampling holes for the neural image only.
    # Numerical depth/visibility and observation masks remain unchanged.
    distance, labels = cv2.distanceTransformWithLabels(255-observed, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    coordinates = np.flatnonzero(observed)
    if not len(coordinates):
        return image, observed, observed.copy()
    fill = np.flatnonzero((observed == 0) & (distance <= 2))
    flat = image.reshape(-1, 3)
    flat[fill] = flat[coordinates[labels.ravel()[fill]-1]]
    usable = observed.copy()
    usable.ravel()[fill] = 255
    return image, observed, usable


def preprocess_joint(rgb_aligned, thermal, ablation="joint"):
    if rgb_aligned.shape != (512, 640, 3) or thermal.shape != (512, 640):
        raise ValueError("Joint inputs must share the 640x512 thermal image domain")
    rgb = rgb_aligned[:, :, ::-1]
    heat = np.repeat(thermal[:, :, None], 3, axis=2)
    if ablation == "thermal_flat":
        heat = np.full_like(heat, int(np.mean(heat)))
    elif ablation == "rgb_flat":
        rgb = np.broadcast_to(np.mean(rgb, axis=(0, 1)).astype(np.uint8), rgb.shape)
    elif ablation != "joint":
        raise ValueError("Unknown modality ablation")
    tensors = {}
    for name, value in [("rgb", rgb), ("thermal", heat)]:
        padded = np.full((640, 640, 3), 114, np.uint8)
        padded[64:576] = value
        tensors[name] = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)/255.
    return tensors


def infer_joint(detector, tensors):
    started = time.perf_counter()
    if detector["backend"] == "tensorrt":
        from multimodal_perception.tensorrt_inference import infer_engine_multi, infer_engine_bound
        output, timing = infer_engine_bound(detector["trt"]) if tensors is None else infer_engine_multi(detector["trt"], tensors)
        raw = output["predictions"]
        detector["metadata"]["cuda_graph"] = bool(detector["trt"].get("graph_instance") is not None)
    else:
        raw = detector["session"].run(["predictions"], tensors)[0]
        timing = {"model_compute_ms": (time.perf_counter()-started)*1000}
    timing["inference_ms"] = (time.perf_counter()-started)*1000
    return raw, timing


def decode_joint(raw, classes, threshold=.25, nms_iou=.45, candidate_subset=False):
    shape_ok = raw.ndim == 3 and raw.shape[0] == 1 and raw.shape[2] == 11 and raw.shape[1] <= 102000 if candidate_subset else raw.shape == (1, 102000, 11)
    if not shape_ok or not np.isfinite(raw).all():
        raise ValueError("Invalid joint detector output")
    probabilities = raw[0, :, 4:5] * raw[0, :, 5:]
    category = probabilities.argmax(axis=1)
    scores = probabilities[np.arange(len(category)), category]
    keep = scores >= threshold
    prediction, category, scores = raw[0, keep], category[keep], scores[keep]
    boxes = np.column_stack((prediction[:, :2]-prediction[:, 2:4]/2, prediction[:, :2]+prediction[:, 2:4]/2))
    boxes[:, [1, 3]] -= 64
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, 640)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, 512)
    detections = []
    for class_id in np.unique(category):
        candidates = np.flatnonzero(category == class_id)
        selected = nonmaximum_suppression(boxes[candidates], scores[candidates], nms_iou)
        for local in selected:
            i = candidates[local]
            if boxes[i, 2] <= boxes[i, 0] or boxes[i, 3] <= boxes[i, 1]:
                continue
            detections.append({"class_id": int(class_id), "class": classes[class_id], "score": float(scores[i]),
                               "bbox_xyxy": boxes[i].tolist(), "bbox_domain": "thermal_left_rectified",
                               "recognition_inputs": ["rgb", "thermal"], "model": "LCAFNet_M3FD"})
    detections.sort(key=lambda d: -d["score"])
    return detections


def attach_joint_geometry(detections, fusion, calibration, min_points=20, max_depth_m=None):
    uv = fusion["thermal_pixels"]
    for identity, detection in enumerate(detections, 1):
        a, b, c, d = detection["bbox_xyxy"]
        inside = (uv[:, 0] >= a) & (uv[:, 0] < c) & (uv[:, 1] >= b) & (uv[:, 1] < d)
        original_count = int(inside.sum())
        if max_depth_m is not None:
            inside &= fusion["points"][:, 2] <= max_depth_m
        count = int(inside.sum())
        detection["id"] = identity
        detection["position"] = {"point_m": None, "frame": calibration["rgb"]["frame"],
                                 "status": "insufficient_depth", "visible_depth_samples": count,
                                 "range_excluded_depth_samples": original_count-count,
                                 "depth_upper_bound_m": max_depth_m}
        detection["rgb_bbox_xyxy"] = None
        detection["thermal"] = {"status": "joint_detection", "support_points": count,
                                "units": "processed_intensity_u8", "bbox_xyxy": detection["bbox_xyxy"]}
        if count >= min_points:
            P = fusion["points"][inside]
            pixel = fusion["rgb_pixels"][inside]
            detection["position"].update(point_m=np.median(P, axis=0).tolist(), status="ok",
                                         depth_mad_m=float(np.median(np.abs(P[:, 2]-np.median(P[:, 2])))))
            detection["rgb_bbox_xyxy"] = np.r_[pixel.min(axis=0), pixel.max(axis=0)+1].tolist()
            detection["thermal"]["intensity_p10_p50_p90"] = np.percentile(fusion["intensity_u8"][inside], [10, 50, 90]).tolist()


def process_joint_sample(sample, detector, calibration, config):
    started = time.perf_counter()
    if not sample["valid"] or sample["ffc"]:
        raise ValueError("Invalid groups must be excluded before joint recognition")
    if config.get("temporal_compensation", False) and config.get("residual_alignment", False):
        raise ValueError("Fixed-clock validation must not use per-frame ECC")
    if "prepared_exposure" in sample:
        exposure_calibration, temporal = sample["prepared_exposure"]
    else:
        exposure_calibration, temporal = calibration_at_exposure(sample, calibration, config.get("temporal_compensation", False))
    motion_finished = time.perf_counter()
    if config.get("registration_backend", "numpy") == "cuda":
        compact = config.get("cuda_compact", False)
        if compact and config.get("residual_alignment", False):
            raise ValueError("GPU resident registration does not support per-frame ECC")
        if "cuda_registration" not in detector:
            from multimodal_perception.cuda_registration import CudaRegistration
            detector["cuda_registration"] = CudaRegistration(compact=compact, mapped_outputs=config.get("cuda_mapped_outputs", False))
        if detector["cuda_registration"].compact != compact:
            raise ValueError("Use a separate detector instance when changing CUDA layout")
        fusion, rgb, observed = detector["cuda_registration"].fuse(sample, exposure_calibration, config["max_time_error_s"])
        if compact:
            usable = fusion.pop("gpu_usable_mask")
        else:
            rgb, observed, usable = fill_registered_holes(rgb, observed)
    elif config.get("registration_backend", "numpy") == "numpy":
        fusion = fuse_thermal(sample, [], exposure_calibration, stride=1, max_time_error_s=config["max_time_error_s"])
        rgb, observed, usable = registered_rgb(sample, fusion)
    else:
        raise ValueError("Unknown registration backend")
    if fusion["status"] == "invalid_group":
        raise ValueError("Unsynchronized modalities must not enter joint recognition")
    geometric = time.perf_counter()
    alignment = {"accepted": False, "delta_uv": [0., 0.], "status": "disabled"}
    if config.get("residual_alignment", False):
        alignment = estimate_residual_shift(rgb, sample["thermal"], observed,
                                            config.get("alignment_min_score", .3),
                                            config.get("alignment_max_shift_px", 30.),
                                            config.get("alignment_min_gain", .01))
        rgb, observed, usable = apply_residual_shift(sample, fusion, rgb, observed, usable, alignment)
    registered = time.perf_counter()
    if config.get("cuda_preprocess", False):
        if "cuda_registration" not in detector or detector["backend"] != "tensorrt":
            raise ValueError("CUDA preprocessing requires CUDA registration and TensorRT")
        detector["cuda_registration"].prepare_engine_inputs(rgb, sample["thermal"], detector["trt"], resident=config.get("cuda_compact", False))
        tensors = None
    else:
        tensors = preprocess_joint(rgb, sample["thermal"])
    prepared = time.perf_counter()
    if config.get("cuda_decode", False):
        if not config.get("cuda_compact", False) or detector["backend"] != "tensorrt":
            raise ValueError("GPU candidate decode requires compact CUDA registration and TensorRT")
        detector["trt"]["output_filter"] = lambda state: detector["cuda_registration"].filter_predictions(state, config["score_threshold"])
    elif detector["backend"] == "tensorrt":
        detector["trt"].pop("output_filter", None)
    raw, timing = infer_joint(detector, tensors)
    inferred = time.perf_counter()
    detections = decode_joint(raw, detector["metadata"]["classes"], config["score_threshold"], config["nms_iou"], config.get("cuda_decode", False))
    attach_joint_geometry(detections, fusion, calibration, config["min_depth_points"], config.get("max_position_depth_m"))
    for detection in detections:
        detection["position"]["time_alignment_status"] = temporal["status"]
    finished = time.perf_counter()
    fusion.update(registered_rgb=rgb, rgb_observed_mask=observed, rgb_usable_mask=usable, residual_alignment=alignment,
                  time_alignment=temporal)
    timing.update(registration_ms=(registered-started)*1000, preprocess_ms=(prepared-registered)*1000,
                  rig_motion_ms=sample.get("preparation_timing", {}).get("rig_motion_ms", (motion_finished-started)*1000),
                  geometric_registration_ms=(geometric-motion_finished)*1000, residual_alignment_ms=(registered-geometric)*1000,
                  postprocess_geometry_ms=(finished-inferred)*1000, processing_ms=(finished-started)*1000)
    return detections, fusion, timing
