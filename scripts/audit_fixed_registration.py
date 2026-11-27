"""Audit a rigid RGB/thermal pair without any per-frame image alignment."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from multimodal_perception.dataset import open_dataset, load_frame
from multimodal_perception.thermal_fusion import load_fusion_calibration


def temporal_motion(first, second):
    corners = cv2.goodFeaturesToTrack(first, 500, .015, 12)
    if corners is None or len(corners) < 15:
        return {"count": 0}
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(first, second, corners, None, winSize=(25, 25), maxLevel=4)
    reverse, back_status, _ = cv2.calcOpticalFlowPyrLK(second, first, tracked, None, winSize=(25, 25), maxLevel=4)
    good = status.ravel().astype(bool) & back_status.ravel().astype(bool)
    good &= np.linalg.norm(reverse[:, 0]-corners[:, 0], axis=1) < .75
    good &= (tracked[:, 0, 0] >= 0) & (tracked[:, 0, 0] < first.shape[1])
    good &= (tracked[:, 0, 1] >= 0) & (tracked[:, 0, 1] < first.shape[0])
    a, b = corners[good, 0], tracked[good, 0]
    if len(a) < 15:
        return {"count": len(a)}
    flow = b-a
    affine, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.)
    return {"count": len(a), "flow_p50_xy": np.median(flow, axis=0).tolist(),
            "flow_magnitude_p50_p95": np.percentile(np.linalg.norm(flow, axis=1), [50, 95]).tolist(),
            "affine": affine.tolist() if affine is not None else None}


def stereo_matches(left, right, depth, fx_baseline):
    detector = cv2.ORB_create(nfeatures=3500, fastThreshold=12)
    kl, dl = detector.detectAndCompute(left, None)
    kr, dr = detector.detectAndCompute(right, None)
    if dl is None or dr is None:
        return {"matches": 0}, []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    forward, reverse = matcher.knnMatch(dl, dr, k=2), matcher.knnMatch(dr, dl, k=2)
    reverse_good = {a.queryIdx: a.trainIdx for pair in reverse if len(pair) == 2
                    for a, b in [pair] if a.distance < .72*b.distance}
    records = []
    for pair in forward:
        if len(pair) != 2:
            continue
        a, b = pair
        if a.distance >= .72*b.distance or reverse_good.get(a.trainIdx) != a.queryIdx:
            continue
        uv, right_uv = np.array(kl[a.queryIdx].pt), np.array(kr[a.trainIdx].pt)
        disparity = uv[0]-right_uv[0]
        if not 1 < disparity < 220 or abs(uv[1]-right_uv[1]) > 2:
            continue
        x, y = np.rint(uv).astype(int)
        predicted = float(depth[y, x])
        if not np.isfinite(predicted) or predicted <= 0:
            continue
        independent = fx_baseline/disparity
        records.append({"rgb_uv": uv.tolist(), "right_uv": right_uv.tolist(), "disparity_px": float(disparity),
                        "published_depth_m": predicted, "stereo_depth_m": float(independent),
                        "published_over_stereo": float(predicted/independent)})
    ratios = np.array([r["published_over_stereo"] for r in records])
    return {"matches": len(records), "published_over_stereo_p10_p50_p90": np.percentile(ratios, [10, 50, 90]).tolist() if len(ratios) else None}, records


def point_projection(points, pose, K):
    transformed = points @ pose["R"].T + pose["t"]
    return transformed[:, :2]/transformed[:, 2:3]*[K[0, 0], K[1, 1]]+[K[0, 2], K[1, 2]]


def structural_samples(sample, camera):
    gray = cv2.cvtColor(sample["rgb"], cv2.COLOR_BGR2GRAY)
    lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(cv2.GaussianBlur(gray, (5, 5), 1.))[0]
    pixels = []
    if lines is not None:
        for line in lines[:, 0]:
            a, b = line[:2], line[2:]
            if np.linalg.norm(b-a) >= 40:
                pixels.extend(a+np.linspace(.12, .88, 10)[:, None]*(b-a))
    pixels = np.asarray(pixels).reshape(-1, 2)
    integer = np.rint(pixels).astype(int)
    z = sample["depth"][integer[:, 1], integer[:, 0]]
    valid = np.isfinite(z) & (z > .2) & (z < 50)
    pixels, z = pixels[valid], z[valid]
    K = camera["K"]
    rays = np.c_[(pixels-[K[0, 2], K[1, 2]])/[K[0, 0], K[1, 1]], np.ones(len(z))]
    return rays, z


def edge_metric(uv, distance, common):
    if not common.any():
        return None
    coords = uv[common].astype(np.float32)
    values = cv2.remap(distance, coords[:, 0].reshape(-1, 1), coords[:, 1].reshape(-1, 1), cv2.INTER_LINEAR).ravel()
    return {"points": len(values), "mean_clipped_20px": float(np.minimum(values, 20).mean()),
            "p50_p90_px": np.percentile(values, [50, 90]).tolist()}


def audit(output, images):
    cv2.setNumThreads(4)
    cv2.setRNGSeed(42)
    output, images = Path(output), Path(images)
    output.mkdir(parents=True, exist_ok=True); images.mkdir(parents=True, exist_ok=True)
    dataset = open_dataset("data/tartan")
    nominal = load_fusion_calibration("data/tartan", "configs/tartan_thermal_profile.json")
    ba = load_fusion_calibration("data/tartan", "configs/tartan_thermal_profile.json", "configs/tartan_scene_refinement.json")
    K, KT = nominal["rgb"]["K"], nominal["thermal"]["K"]
    H = KT @ np.linalg.inv(K)
    stereo_folder = dataset["sequence"] / "zed_right_rect"
    right_times = np.loadtxt(stereo_folder / "timestamps.txt")
    all_frames, all_matches, previous = [], [], None
    preview = set(range(0, 296, 20)) | {100, 150, 250, 295}
    for index in np.flatnonzero(dataset["valid"]).tolist():
        sample = load_frame(dataset, index)
        gray = cv2.cvtColor(sample["rgb"], cv2.COLOR_BGR2GRAY)
        entry = {"index": index, "rgb_timestamp_s": sample["rgb_timestamp_s"], "thermal_timestamp_s": sample["thermal_timestamp_s"],
                 "thermal_minus_rgb_ms": (sample["thermal_timestamp_s"]-sample["rgb_timestamp_s"])*1000,
                 "right_minus_left_ms": float((right_times[index]-sample["rgb_timestamp_s"])*1000)}
        if previous is not None:
            dt = sample["rgb_timestamp_s"]-previous["rgb_timestamp_s"]
            entry["interval_s"] = dt
            for name, first, second, focal in [("rgb", previous["gray"], gray, K[0, 0]),
                                                ("thermal", previous["thermal"], sample["thermal"], KT[0, 0])]:
                motion = temporal_motion(first, second)
                entry[name+"_motion"] = motion
                if "flow_magnitude_p50_p95" in motion:
                    entry[name+"_clock_skew_motion_px_p50_p95"] = (np.array(motion["flow_magnitude_p50_p95"])/dt*abs(entry["thermal_minus_rgb_ms"])/1000*(KT[0, 0]/focal)).tolist()
        right = cv2.imread(str(stereo_folder/f"{index:08d}.png"), cv2.IMREAD_GRAYSCALE)
        if right is None:
            raise ValueError("Missing stereo reference image")
        stereo, matches = stereo_matches(gray, right, sample["depth"], K[0, 0]*.120022)
        entry["stereo"] = stereo
        all_matches.extend({"index": index, **m} for m in matches)
        rays, depth = structural_samples(sample, nominal["rgb"])
        variants = {}
        for name, calibration, scale in [("cad_depth", nominal, 1.), ("ba_depth", ba, 1.),
                                          ("ba_depth_half", ba, .5), ("ba_depth_double", ba, 2.),
                                          ("ba_rotation_only", ba, 1e6)]:
            variants[name] = point_projection(rays*depth[:, None]*scale, calibration["transform"], KT)
        common = np.ones(len(depth), bool)
        for uv in variants.values():
            common &= np.isfinite(uv).all(axis=1) & (uv[:, 0] > 12) & (uv[:, 0] < 628) & (uv[:, 1] > 12) & (uv[:, 1] < 500)
        edge = cv2.Canny(cv2.GaussianBlur(sample["thermal"], (5, 5), 1.), 20, 50)
        distance = cv2.distanceTransform(255-edge, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        entry["structural_edge_proxy"] = {name: edge_metric(uv, distance, common) for name, uv in variants.items()}
        entry["published_depth_p01_p50_p99"] = np.percentile(sample["depth"][np.isfinite(sample["depth"]) & (sample["depth"] > 0)], [1, 50, 99]).tolist()
        all_frames.append(entry)
        if index in preview:
            rgb = cv2.warpPerspective(sample["rgb"], H, (640, 512))
            heat = cv2.cvtColor(sample["thermal"], cv2.COLOR_GRAY2BGR)
            for panel in [rgb, heat]:
                for x in range(0, 640, 40):
                    cv2.line(panel, (x, 0), (x, 511), (50, 120, 50), 1)
                    cv2.putText(panel, str(x), (x+2, 16), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 255, 0), 1)
                for y in range(40, 512, 40):
                    cv2.line(panel, (0, y), (639, y), (50, 120, 50), 1)
                    cv2.putText(panel, str(y), (2, y-2), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 255, 0), 1)
            canvas = np.hstack([rgb, heat])
            canvas = cv2.copyMakeBorder(canvas, 32, 0, 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(canvas, f"{index}: RGB via K only (NO depth, NO pose) | thermal native", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
            cv2.imwrite(str(images/f"native_pair_{index:04d}.jpg"), canvas)
        previous = {**sample, "gray": gray}
        if index % 25 == 0:
            print(index, stereo, flush=True)
    (output/"fixed_audit_frames.json").write_text(json.dumps(all_frames, indent=2)+"\n")
    cache = images/"independent_stereo_matches.json"
    cache.write_text(json.dumps(all_matches, indent=2)+"\n")
    (output/"independent_stereo_matches.index.json").write_text(json.dumps({
        "path": str(cache), "points": len(all_matches), "bytes": cache.stat().st_size,
        "sha256": hashlib.sha256(cache.read_bytes()).hexdigest()}, indent=2)+"\n")
    print("Audited", len(all_frames), "valid groups", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/jetson/tartan_fixed")
    parser.add_argument("--images", default="results/jetson/tartan_fixed/audit")
    args = parser.parse_args()
    audit(args.output, args.images)
