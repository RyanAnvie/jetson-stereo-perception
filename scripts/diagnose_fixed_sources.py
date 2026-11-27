"""Independent clock-phase and subpixel stereo checks for the fixed rig audit."""
import json
from pathlib import Path

import cv2
import numpy as np

from multimodal_perception.dataset import open_dataset, load_frame


def clock_curve(frames, start=-np.inf, stop=np.inf):
    rgb_time = np.array([r["rgb_timestamp_s"] for r in frames])
    thermal_time = np.array([r["thermal_timestamp_s"] for r in frames])
    thermal_time -= rgb_time[0]; rgb_time -= rgb_time[0]
    rgb = np.array([r.get("rgb_motion", {}).get("flow_p50_xy", [np.nan]*2) for r in frames])[1:]
    thermal = np.array([r.get("thermal_motion", {}).get("flow_p50_xy", [np.nan]*2) for r in frames])[1:]
    rgb = rgb/np.diff(rgb_time)[:, None]*(350.0586333355143/369.431)
    thermal = thermal/np.diff(thermal_time)[:, None]
    rgb_time = (rgb_time[1:]+rgb_time[:-1])/2
    thermal_time = (thermal_time[1:]+thermal_time[:-1])/2
    a, b = np.isfinite(rgb).all(axis=1), np.isfinite(thermal).all(axis=1)
    records = []
    for correction in np.arange(-.25, .2501, .005):
        query = thermal_time[b]+correction
        values = np.column_stack([np.interp(query, rgb_time[a], rgb[a, axis]) for axis in range(2)])
        valid = (query > rgb_time[a].min()) & (query < rgb_time[a].max())
        valid &= (thermal_time[b] > start) & (thermal_time[b] < stop)
        correlation = [float(np.corrcoef(values[valid, axis], thermal[b][valid, axis])[0, 1]) for axis in range(2)]
        records.append({"thermal_clock_correction_s": round(float(correction), 6), "motion_intervals": int(valid.sum()),
                        "correlation_xy": correlation, "mean_correlation": float(np.mean(correlation))})
    return {"best": max(records, key=lambda r: r["mean_correlation"]),
            "zero": min(records, key=lambda r: abs(r["thermal_clock_correction_s"])), "curve": records}


def diagnose():
    cv2.setNumThreads(4)
    folder = Path("reports/jetson/tartan_fixed")
    frames = json.loads((folder/"fixed_audit_frames.json").read_text())
    phase = {"whole_sequence": clock_curve(frames), "temporal_blocks": [
        {"interval_s": [a, b], **clock_curve(frames, a+.2, b-.2)} for a, b in [(0, 12), (12, 24), (24, 36)]]}
    phase["interpretation"] = "Independent within-modality image velocity phase; constant effective relative delay, not hardware clock ground truth. Positive correction means thermal appearance is later than its released timestamp relative to RGB. The phase estimates do not fit any cross-modal per-frame image shift."
    offsets = np.abs([r["thermal_minus_rgb_ms"] for r in frames])
    phase["reported_timestamp_skew_ms_p50_p95_max"] = np.percentile(offsets, [50, 95, 100]).tolist()
    for modality in ["rgb", "thermal"]:
        values = [r[modality+"_clock_skew_motion_px_p50_p95"][1] for r in frames if modality+"_clock_skew_motion_px_p50_p95" in r]
        phase[modality+"_reported_skew_motion_bound_px_p50_p95_max"] = np.percentile(values, [50, 95, 100]).tolist()
    (folder/"clock_diagnosis.json").write_text(json.dumps(phase, indent=2)+"\n")
    dataset = open_dataset("data/tartan")
    cache = json.loads((folder/"independent_stereo_matches.index.json").read_text())
    matches = json.loads(Path(cache["path"]).read_text())
    selected = list(range(0, 296, 12))+[295]
    evidence = []
    for index in selected:
        pairs = [r for r in matches if r["index"] == index]
        if not pairs: continue
        sample = load_frame(dataset, index)
        left = cv2.cvtColor(sample["rgb"], cv2.COLOR_BGR2GRAY)
        right = cv2.imread(str(dataset["sequence"]/"zed_right_rect"/f"{index:08d}.png"), cv2.IMREAD_GRAYSCALE)
        a = np.float32([p["rgb_uv"] for p in pairs]).reshape(-1, 1, 2)
        initial = np.float32([p["right_uv"] for p in pairs]).reshape(-1, 1, 2)
        b, good, _ = cv2.calcOpticalFlowPyrLK(left, right, a, initial.copy(), winSize=(15, 15), maxLevel=0,
                                            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
                                            criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, .001))
        back, valid, _ = cv2.calcOpticalFlowPyrLK(right, left, b, a.copy(), winSize=(15, 15), maxLevel=0,
                                               flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        keep = good.ravel().astype(bool) & valid.ravel().astype(bool)
        keep &= (np.linalg.norm(back[:, 0]-a[:, 0], axis=1) < .4) & (np.linalg.norm(b[:, 0]-initial[:, 0], axis=1) < 2.5)
        keep &= np.abs(b[:, 0, 1]-a[:, 0, 1]) < 1.
        disparity = a[:, 0, 0]-b[:, 0, 0]
        keep &= disparity > 3
        for j in np.flatnonzero(keep):
            u, v = a[j, 0]; ur, vr = b[j, 0]
            if min(u, v, ur, vr) < 8 or max(u, ur) >= 952 or max(v, vr) >= 532: continue
            x = cv2.getRectSubPix(left, (11, 11), (float(u), float(v))).astype(float).ravel()
            y = cv2.getRectSubPix(right, (11, 11), (float(ur), float(vr))).astype(float).ravel()
            if min(x.std(), y.std()) < 8: continue
            ncc = float(np.corrcoef(x, y)[0, 1])
            if ncc < .90: continue
            z = pairs[j]["published_depth_m"]
            independent = 369.431*.120022/float(disparity[j])
            evidence.append({"index": index, "rgb_uv": [float(u), float(v)], "ncc": ncc,
                             "measured_disparity_px": float(disparity[j]), "published_disparity_px": 369.431*.120022/z,
                             "published_depth_m": z, "independent_stereo_depth_m": independent,
                             "published_over_stereo": z/independent,
                             "disparity_abs_error_px": abs(float(disparity[j])-369.431*.120022/z)})
        print("stereo subpixel", index, "total accepted", len(evidence), flush=True)
    bins = []
    for lo, hi in [(0, 5), (5, 15), (15, 50), (50, 1e12)]:
        rows = [e for e in evidence if lo < e["published_depth_m"] <= hi]
        bins.append({"published_depth_range_m": [lo, hi], "points": len(rows),
                     "published_over_stereo_p10_p50_p90": np.percentile([r["published_over_stereo"] for r in rows], [10, 50, 90]).tolist() if rows else None,
                     "disparity_abs_error_p50_p95_px": np.percentile([r["disparity_abs_error_px"] for r in rows], [50, 95]).tolist() if rows else None})
    result = {"sample_indices": selected, "points": len(evidence), "bins": bins, "records": evidence,
              "scope": "Independent ORB stereo seeds refined by LK, forward-back <0.4px, vertical residual <1px, disparity >3px, patch NCC >0.90 and textured patches. Published depth is not used to select matches. Stereo remains an image-derived reference, not surveyed metric ground truth; boundary pixels can mix surfaces."}
    (folder/"depth_diagnosis.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"clock_best": phase["whole_sequence"]["best"], "depth_bins": bins}, indent=2))


if __name__ == "__main__":
    diagnose()
