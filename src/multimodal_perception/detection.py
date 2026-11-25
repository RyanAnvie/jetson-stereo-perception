import hashlib
from pathlib import Path
import time

import cv2
import numpy as np


COCO_CLASSES = ("person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,"
                "fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,"
                "elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,"
                "skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,"
                "tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,"
                "orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,"
                "dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,"
                "toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush").split(",")


def load_detector(path, threads=4):
    import onnxruntime as ort

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Download YOLOX Nano first: python scripts/download_model.py")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    if input_info.shape != [1, 3, 416, 416]:
        raise ValueError(f"Expected YOLOX Nano 416x416, received {input_info.shape}")
    grids = []
    strides = []
    for stride in [8, 16, 32]:
        v, u = np.mgrid[0:416 // stride, 0:416 // stride]
        grid = np.column_stack((u.ravel(), v.ravel()))
        grids.append(grid)
        strides.append(np.full((len(grid), 1), stride))
    return {"session": session, "input_name": input_info.name,
            "grid": np.concatenate(grids), "strides": np.concatenate(strides),
            "metadata": {"name": "YOLOX Nano", "release": "0.1.1rc0", "input_shape": [1, 3, 416, 416],
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "precision": "float32",
                         "runtime": ort.__version__, "provider": "CPUExecutionProvider", "threads": threads,
                         "source": "https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0",
                         "license": "Apache-2.0 (upstream project)", "classes": "COCO 80; person=0",
                         "preprocessing": "BGR 0..255 float32 NCHW; aspect-preserving resize; top-left pad=114"}}


def preprocess(image):
    height, width = image.shape[:2]
    ratio = min(416 / height, 416 / width)
    resized = cv2.resize(image, (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_LINEAR)
    padded = np.full((416, 416, 3), 114, dtype=np.uint8)
    padded[:resized.shape[0], :resized.shape[1]] = resized
    tensor = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)
    return tensor, ratio


def nonmaximum_suppression(boxes, scores, threshold):
    order = np.argsort(-scores, kind="stable")
    kept = []
    while len(order):
        first = int(order[0])
        kept.append(first)
        rest = order[1:]
        intersection_min = np.maximum(boxes[first, :2], boxes[rest, :2])
        intersection_max = np.minimum(boxes[first, 2:], boxes[rest, 2:])
        intersection_size = np.maximum(intersection_max - intersection_min, 0)
        intersection = intersection_size[:, 0] * intersection_size[:, 1]
        area_first = np.prod(np.maximum(boxes[first, 2:] - boxes[first, :2], 0))
        area_rest = np.prod(np.maximum(boxes[rest, 2:] - boxes[rest, :2], 0), axis=1)
        union = area_first + area_rest - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = rest[iou <= threshold]
    return kept


def detect(image, detector, score_threshold=0.30, nms_iou=0.45):
    start = time.perf_counter()
    tensor, ratio = preprocess(image)
    prepared = time.perf_counter()
    output = detector["session"].run(None, {detector["input_name"]: tensor})[0]
    inferred = time.perf_counter()
    if output.shape != (1, 3549, 85) or not np.isfinite(output).all():
        raise ValueError(f"Invalid YOLOX output: {output.shape}")
    prediction = output[0].copy()
    centers = (prediction[:, :2] + detector["grid"]) * detector["strides"]
    sizes = np.exp(prediction[:, 2:4]) * detector["strides"]
    boxes = np.column_stack((centers - sizes / 2, centers + sizes / 2)) / ratio
    scores = prediction[:, 4:5] * prediction[:, 5:]
    classes = np.argmax(scores, axis=1)
    best_scores = scores[np.arange(len(scores)), classes]
    selected = best_scores >= score_threshold
    boxes, best_scores, classes = boxes[selected], best_scores[selected], classes[selected]
    detections = []
    for class_id in np.unique(classes):
        indices = np.flatnonzero(classes == class_id)
        keep = nonmaximum_suppression(boxes[indices], best_scores[indices], nms_iou)
        for local_index in keep:
            index = indices[local_index]
            box = boxes[index].copy()
            box[[0, 2]] = np.clip(box[[0, 2]], 0, image.shape[1])
            box[[1, 3]] = np.clip(box[[1, 3]], 0, image.shape[0])
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            detections.append({"class_id": int(class_id), "class": COCO_CLASSES[class_id],
                               "score": float(best_scores[index]), "bbox_xyxy": box.tolist()})
    finished = time.perf_counter()
    timing = {"preprocess_ms": (prepared-start)*1000, "inference_ms": (inferred-prepared)*1000,
              "postprocess_ms": (finished-inferred)*1000}
    return detections, timing
