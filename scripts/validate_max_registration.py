"""Validate every compact/mapped CUDA image and point against the original CPU path."""
import json
from pathlib import Path

import cv2
import numpy as np

from multimodal_perception.cuda_registration import CudaRegistration
from multimodal_perception.dataset import open_dataset
from multimodal_perception.joint_prefetch import prepared_samples
from multimodal_perception.thermal_fusion import load_fusion_calibration, fuse_thermal
from validate_fast_registration import original_raster


def validate():
    cv2.setNumThreads(4)
    dataset = open_dataset("data/tartan")
    calibration = load_fusion_calibration("data/tartan", "configs/tartan_thermal_profile.json", "configs/tartan_fixed_refinement.json")
    indices = np.flatnonzero(dataset["valid"]).tolist()
    prepared = prepared_samples(dataset, indices, calibration, True, 4, 2)
    gpu = CudaRegistration(compact=True, mapped_outputs=True)
    records = []
    try:
        for sample in prepared:
            cal, _ = sample["prepared_exposure"]
            cpu = fuse_thermal(sample, [], cal, stride=1)
            rgb, observed, usable = original_raster(sample, cpu)
            fast, new_rgb, new_observed = gpu.fuse(sample, cal)
            equal = {key: np.array_equal(cpu[key], fast[key]) for key in ["points", "rgb_pixels", "thermal_pixels", "intensity_u8"]}
            equal.update(rgb_image=np.array_equal(rgb, new_rgb), observed=np.array_equal(observed, new_observed),
                         usable=np.array_equal(usable, fast["gpu_usable_mask"]), sampled_depth_points=cpu["sampled_depth_points"] == fast["sampled_depth_points"])
            records.append({"index": sample["index"], "exact": equal, "visible_points": len(cpu["points"])})
            if sample["index"] % 25 == 0:
                print(records[-1], flush=True)
    finally:
        prepared.close()
        gpu.close()
    failures = [r["index"] for r in records if not all(r["exact"].values())]
    report = {"frames": len(records), "failed_frames": failures, "all_images_points_and_masks_exact": not failures,
              "geometry": "FP64; GPU stable compaction and radius-2 fill; mapped host outputs",
              "records": records}
    Path("reports/jetson/tartan_max/registration_equivalence.json").write_text(json.dumps(report, indent=2)+"\n")
    assert len(records) == len(indices) and not failures
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    validate()
