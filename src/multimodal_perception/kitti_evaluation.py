"""Offline reference checks; no labels or LiDAR enter RGB stereo inference."""

import numpy as np

from multimodal_perception.geometry import backproject
from multimodal_perception.kitti import points_in_tracklet, project_tracklet
from multimodal_perception.stereo import project_lidar


def box_iou(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    overlap = np.maximum(np.minimum(first[2:], second[2:])-np.maximum(first[:2], second[:2]), 0)
    intersection = float(np.prod(overlap))
    union = np.prod(first[2:]-first[:2]) + np.prod(second[2:]-second[:2]) - intersection
    return intersection/union if union > 0 else 0.0


def match_detections(detections, labels, class_mapping, threshold):
    candidates = []
    for di, detection in enumerate(detections):
        for gi, label in enumerate(labels):
            if class_mapping.get(label["class"]) != detection["class"]:
                continue
            iou = box_iou(detection["bbox_xyxy"], label["bbox_xyxy"])
            if iou >= threshold:
                candidates.append((iou, di, gi))
    candidates.sort(reverse=True)
    used_detections = set()
    used_labels = set()
    matches = []
    for iou, di, gi in candidates:
        if di not in used_detections and gi not in used_labels:
            matches.append({"detection_index": di, "label_index": gi, "iou": iou})
            used_detections.add(di)
            used_labels.add(gi)
    return matches


def error_metrics(errors, reference_z):
    errors = np.asarray(errors, dtype=float)
    reference_z = np.asarray(reference_z, dtype=float)
    if not len(errors):
        return {"count": 0, "mae_m": None, "rmse_m": None, "bias_m": None,
                "median_abs_error_m": None, "p95_abs_error_m": None, "mean_abs_relative_error": None}
    if not np.isfinite(errors).all() or not np.isfinite(reference_z).all() or np.any(reference_z <= 0):
        raise ValueError("Metrics require explicitly selected finite errors and positive reference depths")
    absolute = np.abs(errors)
    return {"count": len(errors), "mae_m": float(absolute.mean()),
            "rmse_m": float(np.sqrt(np.mean(errors**2))), "bias_m": float(errors.mean()),
            "median_abs_error_m": float(np.median(absolute)), "p95_abs_error_m": float(np.percentile(absolute, 95)),
            "mean_abs_relative_error": float(np.mean(absolute/reference_z))}


def evaluate_kitti_frame(sample, depth, detections, calibration, config):
    lidar = project_lidar(sample["lidar_xyzi"], calibration)
    uv = lidar["pixels_uv"]
    reference = lidar["points_cam2_m"]
    estimated_z = depth[uv[:, 1], uv[:, 0]]
    supported = np.isfinite(estimated_z) & (estimated_z > 0)
    errors = estimated_z[supported]-reference[supported, 2]
    reference_z = reference[supported, 2]
    labels = []
    for tracklet in sample["tracklets"]:
        # Fully in-image and not fully occluded; all exclusions are reported.
        if tracklet["class"] not in config["evaluation_classes"]:
            continue
        if tracklet["occlusion"] not in [0, 1] or tracklet["truncation"] != 0:
            continue
        projected = project_tracklet(tracklet, calibration)
        if projected is not None:
            labels.append(projected)
    matches = match_detections(detections, labels, config["evaluation_classes"], config["evaluation_iou"])
    object_rows = []
    object_errors = []
    object_reference_z = []
    object_union = np.zeros(len(uv), dtype=bool)
    for match in matches:
        detection = detections[match["detection_index"]]
        label = labels[match["label_index"]]
        original_points = sample["lidar_xyzi"][lidar["scan_indices"], :3]
        inside = points_in_tracklet(original_points, label)
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        inside &= (uv[:, 0] >= x1) & (uv[:, 0] < x2) & (uv[:, 1] >= y1) & (uv[:, 1] < y2)
        good = inside & supported
        object_union |= good
        row = {"frame_index": sample["index"], "track_id": label["track_id"], "class": label["class"],
               "detection_index": match["detection_index"], "iou": match["iou"],
               "reference_surface_pixels": int(inside.sum()), "valid_stereo_pixels": int(good.sum()),
               "stereo_coverage": float(good.sum()/inside.sum()) if inside.any() else None,
               "status": "ok" if good.any() else "no_common_reference_pixels"}
        if good.any():
            predicted_points, _ = backproject(uv[good], estimated_z[good], calibration["camera"])
            xyz_difference = predicted_points-reference[good]
            errors_object = estimated_z[good]-reference[good, 2]
            row.update(error_metrics(errors_object, reference[good, 2]))
            row["mean_xyz_distance_m"] = float(np.linalg.norm(xyz_difference, axis=1).mean())
            row["median_surface_point_discrepancy_m"] = float(np.linalg.norm(np.median(predicted_points, axis=0)-np.median(reference[good], axis=0)))
        object_rows.append(row)
    object_errors = estimated_z[object_union]-reference[object_union, 2]
    object_reference_z = reference[object_union, 2]
    result = {"index": sample["index"], "lidar_projected_pixels": len(uv), "valid_stereo_pixels": int(supported.sum()),
              "stereo_coverage_at_lidar": float(supported.mean()) if len(supported) else None,
              "source_labels": len(sample["tracklets"]), "eligible_labels": len(labels),
              "matched_detections": len(matches), "depth": error_metrics(errors, reference_z)}
    return {"frame": result, "objects": object_rows, "labels": labels, "matches": matches, "lidar": lidar,
            "errors": errors, "reference_z": reference_z,
            "object_errors": object_errors, "object_reference_z": object_reference_z}


def summarize_evaluation(evaluations):
    errors = np.concatenate([item["errors"] for item in evaluations])
    reference_z = np.concatenate([item["reference_z"] for item in evaluations])
    object_errors = np.concatenate([item["object_errors"] for item in evaluations])
    object_z = np.concatenate([item["object_reference_z"] for item in evaluations])
    projected = sum(item["frame"]["lidar_projected_pixels"] for item in evaluations)
    objects = [row for item in evaluations for row in item["objects"]]
    ranges = []
    # These are reporting bins only; no samples are filtered out of total metrics.
    for low, high in [(0, 10), (10, 20), (20, 40), (40, np.inf)]:
        selected = (reference_z >= low) & (reference_z < high)
        ranges.append({"reference_z_min_m": low, "reference_z_max_m": high if np.isfinite(high) else None,
                       **error_metrics(errors[selected], reference_z[selected])})
    return {"frames": len(evaluations), "depth_all_common_pixels": error_metrics(errors, reference_z),
            "depth_matched_object_surface_pixels": error_metrics(object_errors, object_z),
            "total_projected_lidar_pixels": projected, "stereo_coverage_at_lidar": len(errors)/projected if projected else None,
            "range_bins": ranges, "eligible_label_instances": sum(item["frame"]["eligible_labels"] for item in evaluations),
            "matched_detection_instances": sum(item["frame"]["matched_detections"] for item in evaluations),
            "matched_objects_with_depth_reference": sum(row["valid_stereo_pixels"] > 0 for row in objects),
            "metric_scope": "Stereo depth/3D surface discrepancy at same LiDAR-supported pixels, not object-center error or official KITTI benchmark",
            "reference_limits": "Raw rotating LiDAR scans without motion compensation; sparse sampling, calibration and time skew affect comparisons",
            "inference_uses_reference": False,
            "label_use": "Offline association only. No official precision/recall/mAP: projected raw tracklets are not complete 2D detection labels."}
