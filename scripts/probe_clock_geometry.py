import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from multimodal_perception.dataset import open_dataset, load_frame
from multimodal_perception.thermal_fusion import load_fusion_calibration, fuse_thermal
from multimodal_perception.joint_detection import registered_rgb
from multimodal_perception.camera import rigid_transform
from multimodal_perception.geometry import depth_cloud, thermal_correspondence
from multimodal_perception.rig_motion import estimate_rgb_motion, interpolate_rgb_motion
from multimodal_perception.scene_calibration import project_pose
from audit_fixed_registration import structural_samples, point_projection, edge_metric


def probe():
    cv2.setNumThreads(4)
    dataset = open_dataset("data/tartan")
    calibration = load_fusion_calibration("data/tartan", "configs/tartan_thermal_profile.json")
    ba = load_fusion_calibration("data/tartan", "configs/tartan_thermal_profile.json", "configs/tartan_scene_refinement.json")
    output = Path("reports/jetson/tartan_fixed")
    images = Path("results/jetson/tartan_fixed/clock_probe"); images.mkdir(parents=True, exist_ok=True)
    cache = output/"rgb_motion.json"
    if cache.is_file():
        motions = json.loads(cache.read_text())
    else:
        motions = []
        first = load_frame(dataset, 0)
        for i in range(296):
            second = load_frame(dataset, i+1)
            motion = estimate_rgb_motion(first, second, calibration["rgb"])
            motions.append(motion); first = second
            if i % 25 == 0: print("motion", i, motion["status"], motion.get("inliers"), flush=True)
        cache.write_text(json.dumps(motions, indent=2)+"\n")
    annotations = json.loads(Path("configs/tartan_scene_lines.json").read_text())
    training = []
    for index in annotations["training_frames"]:
        sample = load_frame(dataset, index)
        P, pixels = depth_cloud(sample["depth"], calibration["rgb"], 1)
        selected, uv = thermal_correspondence(P, calibration["transform"], calibration["thermal"], calibration["rgb"]["frame"], True)
        tree = cKDTree(uv)
        for line in annotations["lines"]:
            if line["frame"] != index: continue
            a, b = np.asarray(line["rgb_nominal"])
            distance, nearest = tree.query(a+np.linspace(.1, .9, 8)[:, None]*(b-a))
            points = P[selected[nearest]]
            good = (distance < 2) & (points[:, 2] > .4) & (points[:, 2] < 30)
            ta, tb = np.asarray(line["thermal"]); tangent = tb-ta
            normal = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
            training.append({"index": index, "name": line["name"], "points": points[good], "normal": normal,
                             "offset": -normal@ta, "count": int(good.sum()),
                             "skew": sample["thermal_timestamp_s"]-sample["rgb_timestamp_s"]})
    original = np.r_[np.zeros(3), calibration["transform"]["t"]]
    fits = []
    for lag in [0., .04, .06, .07, .08]:
        def residual(pose):
            values = []
            for record in training:
                if not record["count"]: continue
                P = record["points"]
                if lag > 0:
                    R, t = interpolate_rgb_motion(motions[record["index"]], lag+record["skew"])
                    P = P @ R.T+t
                uv = project_pose(P, pose, calibration["thermal"]["K"])
                values.extend((uv@record["normal"]+record["offset"]) / (4*np.sqrt(record["count"])))
            return np.r_[values, pose[:3]/.05, (pose[3:]-original[3:])/.005]
        fit = least_squares(residual, original, loss="soft_l1", max_nfev=200)
        fits.append({"lag_s": lag, "pose": fit.x.tolist(), "cost": fit.cost, "success": bool(fit.success)})
        print("fit", fits[-1], flush=True)
    (output/"clock_extrinsic_probe_fits.json").write_text(json.dumps(fits, indent=2)+"\n")
    frames = []
    for index in range(296):
        sample = load_frame(dataset, index)
        rays, z = structural_samples(sample, calibration["rgb"])
        P = rays*z[:, None]
        variants = {"old_ba_no_clock": point_projection(P, ba["transform"], calibration["thermal"]["K"])}
        transforms = {}
        for fit in fits:
            pose = np.array(fit["pose"]); R = cv2.Rodrigues(pose[:3])[0]; t = pose[3:]
            lag = fit["lag_s"]
            if lag > 0:
                if not motions[index]["accepted"]: continue
                duration = lag+sample["thermal_timestamp_s"]-sample["rgb_timestamp_s"]
                if duration > motions[index]["dt_s"]: continue
                motion_R, motion_t = interpolate_rgb_motion(motions[index], duration)
                t = R@motion_t+t; R = R@motion_R
            key = f"clock_{lag:.3f}"
            transform = rigid_transform(R, t, calibration["rgb"]["frame"], calibration["thermal"]["frame"])
            transforms[key] = transform
            variants[key] = point_projection(P, transform, calibration["thermal"]["K"])
        common = np.ones(len(P), bool)
        for uv in variants.values(): common &= np.isfinite(uv).all(axis=1)&(uv[:, 0] > 12)&(uv[:, 0] < 628)&(uv[:, 1] > 12)&(uv[:, 1] < 500)
        edges = cv2.Canny(cv2.GaussianBlur(sample["thermal"], (5, 5), 1), 20, 50)
        distance = cv2.distanceTransform(255-edges, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        frames.append({"index": index, "metrics": {key: edge_metric(uv, distance, common) for key, uv in variants.items()}})
        if index in [0, 25, 100, 150, 160, 200, 250, 280, 290, 295]:
            panels = []
            for key, transform in [("old_ba_no_clock", ba["transform"]), ("clock_0.070", transforms.get("clock_0.070", ba["transform"]))]:
                revised = {**calibration, "transform": transform}
                fusion = fuse_thermal(sample, [], revised, stride=1)
                rgb, _, _ = registered_rgb(sample, fusion)
                heat = cv2.applyColorMap(sample["thermal"], cv2.COLORMAP_INFERNO)
                canvas = cv2.addWeighted(rgb, .55, heat, .45, 0)
                canvas = cv2.copyMakeBorder(canvas, 32, 0, 0, 0, cv2.BORDER_CONSTANT)
                cv2.putText(canvas, f"{index} {key}", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
                panels.append(canvas)
            cv2.imwrite(str(images/f"{index:04d}.jpg"), np.hstack(panels))
        if index % 25 == 0: print("evaluate", index, flush=True)
    (output/"clock_extrinsic_probe_frames.json").write_text(json.dumps(frames, indent=2)+"\n")


if __name__ == "__main__":
    probe()
