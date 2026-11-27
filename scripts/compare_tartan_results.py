import argparse
import json
from pathlib import Path

import numpy as np


def compare(reference_path, candidate_path):
    reference = json.loads(Path(reference_path).read_text())
    candidate = json.loads(Path(candidate_path).read_text())
    left = {o["index"]: o for o in reference}
    right = {o["index"]: o for o in candidate}
    differences, box_errors, score_errors, position_errors = [], [], [], []
    pairs, thermal_equal = 0, 0
    for index in sorted(left.keys() & right.keys()):
        a, b = left[index]["detections"], right[index]["detections"]
        if [d["class_id"] for d in a] != [d["class_id"] for d in b]:
            differences.append(index)
            continue
        for first, second in zip(a, b):
            pairs += 1
            box_errors.extend(np.abs(np.array(first["bbox_xyxy"])-second["bbox_xyxy"]).tolist())
            score_errors.append(abs(first["score"]-second["score"]))
            xyz_a, xyz_b = first["position"]["point_m"], second["position"]["point_m"]
            if xyz_a is not None and xyz_b is not None:
                position_errors.append(float(np.linalg.norm(np.array(xyz_a)-xyz_b)))
            elif (xyz_a is None) != (xyz_b is None):
                differences.append(index)
            thermal_equal += first["thermal"] == second["thermal"]
    return {"reference": str(reference_path), "candidate": str(candidate_path),
            "common_groups": len(left.keys() & right.keys()),
            "missing_candidate_indices": sorted(left.keys()-right.keys()),
            "extra_candidate_indices": sorted(right.keys()-left.keys()),
            "class_or_position_validity_mismatch_indices": sorted(set(differences)),
            "paired_detections": pairs, "exact_thermal_association_pairs": thermal_equal,
            "bbox_max_abs_error_px": max(box_errors, default=0), "score_max_abs_error": max(score_errors, default=0),
            "position_max_l2_error_m": max(position_errors, default=0),
            "all_outputs_exact_equal": reference == candidate,
            "matching": "frame index and decoder order; class/count mismatches explicitly reported"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare recorded TartanRGBT fusion outputs")
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
