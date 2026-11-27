"""Exhaustive CUDA / original CPU geometry and registered-image equivalence."""
import json
from pathlib import Path
import time

import cv2
import numpy as np

from multimodal_perception.cuda_registration import CudaRegistration
from multimodal_perception.dataset import open_dataset
from multimodal_perception.joint_detection import fill_registered_holes
from multimodal_perception.joint_prefetch import prepared_samples
from multimodal_perception.thermal_fusion import load_fusion_calibration, fuse_thermal


def original_raster(sample, fusion):
    # Frozen pre-optimization raster/fill algorithm, including OpenCV tie rules.
    uv, pixels = fusion["thermal_pixels"], fusion["rgb_pixels"]
    image = np.full((512, 640, 3), 114, dtype=np.uint8)
    observed = np.zeros((512, 640), dtype=np.uint8)
    if not len(uv):
        return image, observed, observed
    image[uv[:, 1], uv[:, 0]] = sample["rgb"][pixels[:, 1], pixels[:, 0]]
    observed[uv[:, 1], uv[:, 0]] = 255
    distance, labels = cv2.distanceTransformWithLabels(255-observed, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    coordinates = np.column_stack(np.nonzero(observed))
    nearest = coordinates[np.maximum(labels-1, 0)]
    fill = (observed == 0) & (distance <= 2)
    image[fill] = image[nearest[fill, 0], nearest[fill, 1]]
    usable = ((observed > 0) | fill).astype(np.uint8)*255
    return image, observed, usable


def validate():
    cv2.setNumThreads(4)
    dataset = open_dataset("data/tartan")
    calibration = load_fusion_calibration("data/tartan", "configs/tartan_thermal_profile.json", "configs/tartan_fixed_refinement.json")
    indices = np.flatnonzero(dataset["valid"]).tolist()
    prepared = prepared_samples(dataset, indices, calibration, True)
    gpu = CudaRegistration()
    records = []
    try:
        for sample in prepared:
            cal, _ = sample["prepared_exposure"]
            tick = time.perf_counter()
            cpu = fuse_thermal(sample, [], cal, stride=1)
            rgb, observed, usable = original_raster(sample, cpu)
            cpu_ms = (time.perf_counter()-tick)*1000
            tick = time.perf_counter()
            fast, new_rgb, new_observed = gpu.fuse(sample, cal)
            new_rgb, new_observed, new_usable = fill_registered_holes(new_rgb, new_observed)
            gpu_ms = (time.perf_counter()-tick)*1000
            record = {"index": sample["index"], "cpu_ms": cpu_ms, "cuda_ms": gpu_ms,
                      "selected_source_pixels_equal": np.array_equal(cpu["rgb_pixels"], fast["rgb_pixels"]),
                      "thermal_pixels_equal": np.array_equal(cpu["thermal_pixels"], fast["thermal_pixels"]),
                      "intensities_equal": np.array_equal(cpu["intensity_u8"], fast["intensity_u8"]),
                      "registered_image_changed_values": int(np.count_nonzero(rgb != new_rgb)),
                      "observed_changed_values": int(np.count_nonzero(observed != new_observed)),
                      "usable_changed_values": int(np.count_nonzero(usable != new_usable)),
                      "sampled_depth_points_equal": cpu["sampled_depth_points"] == fast["sampled_depth_points"],
                      "xyz_max_abs_error_m": float(np.max(np.abs(cpu["points"]-fast["points"]))) if cpu["points"].shape == fast["points"].shape else None}
            records.append(record)
            if sample["index"] % 25 == 0:
                print(record, flush=True)
    finally:
        prepared.close()
        gpu.close()
    failures = [r["index"] for r in records if not all(r[k] for k in ["selected_source_pixels_equal", "thermal_pixels_equal", "intensities_equal", "sampled_depth_points_equal"])
                or any(r[k] for k in ["registered_image_changed_values", "observed_changed_values", "usable_changed_values"])
                or r["xyz_max_abs_error_m"] is None or r["xyz_max_abs_error_m"] > 1e-9]
    report = {"frames": len(records), "failed_frames": failures,
              "registered_images_bit_exact": not any(r["registered_image_changed_values"] for r in records),
              "xyz_max_abs_error_m": max(r["xyz_max_abs_error_m"] or 0 for r in records),
              "scope": "Every valid frame, unchanged fixed extrinsics and RGB motion. FP64 geometry and exact raster/mask comparison to pre-optimization algorithm, not detection ground truth.",
              "records": records}
    Path("reports/jetson/tartan_fast/registration_equivalence.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2), flush=True)
    assert len(records) == len(indices) and not failures


if __name__ == "__main__":
    validate()
