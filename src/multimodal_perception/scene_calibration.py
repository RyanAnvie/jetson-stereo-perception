"""Local RGB-D / thermal-line bundle adjustment with explicit gauge and priors."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree

from multimodal_perception.camera import rigid_transform
from multimodal_perception.dataset import load_frame, open_dataset
from multimodal_perception.geometry import backproject, depth_cloud, thermal_correspondence
from multimodal_perception.thermal_fusion import load_fusion_calibration


def project_pose(points, pose, K):
    transformed = points @ cv2.Rodrigues(pose[:3])[0].T + pose[3:]
    z = np.maximum(transformed[:, 2], 1e-4)
    return transformed[:, :2] / z[:, None] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]


def collect_scene_windows(dataset, calibration, annotations, thermal_clock_correction_s=0.):
    windows = []
    K = calibration["rgb"]["K"]
    for index in annotations["training_frames"]:
        sample = load_frame(dataset, index)
        points, pixels = depth_cloud(sample["depth"], calibration["rgb"], 1)
        selected, uv = thermal_correspondence(points, calibration["transform"], calibration["thermal"],
                                              calibration["rgb"]["frame"], verified=True)
        tree = cKDTree(uv)
        landmark_points, landmark_pixels, thermal_observations = [], [], []
        for line_id, line in enumerate(annotations["lines"]):
            if line["frame"] != index:
                continue
            a, b = np.asarray(line["rgb_nominal"], dtype=float)
            queries = a + np.linspace(.1, .9, 8)[:, None] * (b-a)
            distance, nearest = tree.query(queries)
            source = selected[nearest]
            valid = (distance < 2) & (points[source, 2] > .4) & (points[source, 2] < 30)
            ta, tb = np.asarray(line["thermal"], dtype=float)
            tangent = tb-ta
            normal = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
            for source_index in source[valid]:
                thermal_observations.append({"point": len(landmark_points), "normal": normal,
                                             "offset": float(-normal @ ta), "line_id": line_id,
                                             "sigma": annotations["annotation_sigma_px"] * np.sqrt(max(1, valid.sum()))})
                landmark_points.append(points[source_index])
                landmark_pixels.append(pixels[source_index])
        gray = cv2.cvtColor(sample["rgb"], cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=150, qualityLevel=.02, minDistance=12)
        if corners is not None:
            corners = corners[:, 0]
            integer = np.round(corners).astype(int)
            depths = sample["depth"][integer[:, 1], integer[:, 0]]
            corner_points, valid = backproject(corners, depths, calibration["rgb"])
            valid &= (depths > .4) & (depths < 30)
            landmark_points.extend(corner_points[valid])
            landmark_pixels.extend(corners[valid])
        P = np.array(landmark_points, dtype=float)
        uv0 = np.array(landmark_pixels, dtype=np.float32)
        if len(P) < 12:
            raise ValueError(f"Too few scene points at frame {index}")
        views = [{"index": index, "pose": np.zeros(6), "point_indices": np.arange(len(P)), "uv": uv0}]
        neighbors = [index-2, index+1, index+2] if thermal_clock_correction_s else [index-2, index+2]
        for adjacent in neighbors:
            if adjacent < 0 or adjacent >= len(dataset["valid"]) or not dataset["valid"][adjacent]:
                continue
            neighbor = load_frame(dataset, adjacent)
            next_gray = cv2.cvtColor(neighbor["rgb"], cv2.COLOR_BGR2GRAY)
            tracked, status, _ = cv2.calcOpticalFlowPyrLK(gray, next_gray, uv0.reshape(-1, 1, 2), None,
                                                        winSize=(31, 31), maxLevel=3)
            backward, reverse_status, _ = cv2.calcOpticalFlowPyrLK(next_gray, gray, tracked, None,
                                                                 winSize=(31, 31), maxLevel=3)
            valid = status.ravel().astype(bool) & reverse_status.ravel().astype(bool)
            valid &= np.linalg.norm(backward[:, 0]-uv0, axis=1) < 1.
            ids = np.flatnonzero(valid)
            if len(ids) < 12:
                continue
            ok, rotation, translation, inliers = cv2.solvePnPRansac(
                P[ids], tracked[ids, 0], K, None, iterationsCount=150,
                reprojectionError=2.5, confidence=.999, flags=cv2.SOLVEPNP_EPNP)
            if not ok or inliers is None or len(inliers) < 12:
                continue
            ids = ids[inliers.ravel()]
            views.append({"index": adjacent, "pose": np.r_[rotation.ravel(), translation.ravel()],
                          "point_indices": ids, "uv": tracked[ids, 0]})
        if len(views) < 2:
            raise ValueError(f"No reliable temporal RGB pose constraint at frame {index}")
        window = {"anchor": index, "initial_points": P, "views": views, "thermal": thermal_observations}
        if thermal_clock_correction_s:
            exposure = sample["thermal_timestamp_s"]+thermal_clock_correction_s-sample["rgb_timestamp_s"]
            candidates = [v for v in views[1:] if v["index"] > index
                          and dataset["zed_left_rect"]["actual"][v["index"]]-sample["rgb_timestamp_s"] >= exposure]
            if exposure < 0 or not candidates:
                raise ValueError("No RGB pose bracket for the corrected thermal exposure")
            view = min(candidates, key=lambda v: v["index"])
            window["thermal_view"] = view
            window["thermal_alpha"] = float(exposure/(dataset["zed_left_rect"]["actual"][view["index"]]-sample["rgb_timestamp_s"]))
        windows.append(window)
    return windows


def fit_scene_bundle(windows, calibration, sigma_px):
    K_rgb, K_thermal = calibration["rgb"]["K"], calibration["thermal"]["K"]
    initial_extrinsic = np.r_[np.zeros(3), calibration["transform"]["t"]]

    def thermal_residual(pose):
        values = []
        for window in windows:
            P = window["initial_points"]
            if "thermal_view" in window:
                moving = window["thermal_view"]["pose"]*window["thermal_alpha"]
                P = P @ cv2.Rodrigues(moving[:3])[0].T+moving[3:]
            uv = project_pose(P, pose, K_thermal)
            values.extend((uv[o["point"]] @ o["normal"] + o["offset"]) / o["sigma"] for o in window["thermal"])
        return np.r_[values, pose[:3]/.05, (pose[3:]-initial_extrinsic[3:])/.005]

    seed = least_squares(thermal_residual, initial_extrinsic, loss="soft_l1", f_scale=1.,
                         bounds=([-0.1]*3+[-.3]*3, [.1]*3+[.3]*3), max_nfev=150)
    initial = seed.x.tolist()
    for window in windows:
        window["point_start"] = len(initial)
        initial.extend(window["initial_points"].ravel())
        for view in window["views"][1:]:
            view["pose_start"] = len(initial)
            initial.extend(view["pose"])
    initial = np.asarray(initial)

    def residual(parameters):
        result = []
        for window in windows:
            count = len(window["initial_points"])
            start = window["point_start"]
            P = parameters[start:start+3*count].reshape(-1, 3)
            for view in window["views"]:
                pose = np.zeros(6) if "pose_start" not in view else parameters[view["pose_start"]:view["pose_start"]+6]
                ids = view["point_indices"]
                result.extend((project_pose(P[ids], pose, K_rgb)-view["uv"]).ravel())
            # Depth anchors fix metric scale. Each window's central RGB pose is fixed.
            result.extend((P[:, 2]-window["initial_points"][:, 2]) / np.maximum(.03, .05*window["initial_points"][:, 2]))
            thermal_points = P
            if "thermal_view" in window:
                offset = window["thermal_view"]["pose_start"]
                moving = parameters[offset:offset+6]*window["thermal_alpha"]
                thermal_points = P @ cv2.Rodrigues(moving[:3])[0].T+moving[3:]
            thermal_uv = project_pose(thermal_points, parameters[:6], K_thermal)
            result.extend((thermal_uv[o["point"]] @ o["normal"] + o["offset"]) / o["sigma"] for o in window["thermal"])
            for view in window["views"][1:]:
                offset = view["pose_start"]
                result.extend((parameters[offset:offset+6]-view["pose"]) / [.02, .02, .02, .05, .05, .05])
        result.extend(parameters[:3]/.05)
        result.extend((parameters[3:6]-initial_extrinsic[3:])/.005)
        return np.asarray(result)

    sparsity = lil_matrix((len(residual(initial)), len(initial)), dtype=int)
    row = 0
    for window in windows:
        count, start = len(window["initial_points"]), window["point_start"]
        for view in window["views"]:
            for point in view["point_indices"]:
                sparsity[row:row+2, start+3*point:start+3*point+3] = 1
                if "pose_start" in view:
                    offset = view["pose_start"]
                    sparsity[row:row+2, offset:offset+6] = 1
                row += 2
        for point in range(count):
            sparsity[row, start+3*point+2] = 1
            row += 1
        for observation in window["thermal"]:
            point = observation["point"]
            sparsity[row, start+3*point:start+3*point+3] = 1
            sparsity[row, :6] = 1
            if "thermal_view" in window:
                offset = window["thermal_view"]["pose_start"]
                sparsity[row, offset:offset+6] = 1
            row += 1
        for view in window["views"][1:]:
            offset = view["pose_start"]
            sparsity[row:row+6, offset:offset+6] = 1
            row += 6
    sparsity[row:row+6, :6] = 1
    lower, upper = np.full(len(initial), -np.inf), np.full(len(initial), np.inf)
    lower[:6], upper[:6] = [-.1]*3+[-.3]*3, [.1]*3+[.3]*3
    for window in windows:
        start, count = window["point_start"], len(window["initial_points"])
        lower[start+2:start+3*count:3] = .1
    fit = least_squares(residual, initial, jac_sparsity=sparsity.tocsr(), bounds=(lower, upper),
                        loss="soft_l1", f_scale=1., x_scale="jac", max_nfev=70, ftol=1e-5, xtol=1e-6)
    summaries = []
    for window in windows:
        start, count = window["point_start"], len(window["initial_points"])
        P = fit.x[start:start+3*count].reshape(-1, 3)
        poses = []
        for view in window["views"]:
            pose = np.zeros(6) if "pose_start" not in view else fit.x[view["pose_start"]:view["pose_start"]+6]
            ids = view["point_indices"]
            error = np.linalg.norm(project_pose(P[ids], pose, K_rgb)-view["uv"], axis=1)
            R = cv2.Rodrigues(pose[:3])[0]
            extrinsic_R = cv2.Rodrigues(fit.x[:3])[0]
            poses.append({"frame": view["index"], "rgb_from_window_rvec_t": pose.tolist(),
                          "thermal_from_window_R": (extrinsic_R @ R).tolist(),
                          "thermal_from_window_t_m": (extrinsic_R @ pose[3:]+fit.x[3:6]).tolist(),
                          "rgb_track_points": len(ids), "rgb_reprojection_p50_p95_px": np.percentile(error, [50, 95]).tolist()})
        summaries.append({"anchor_frame": window["anchor"], "landmarks": count,
                          "thermal_line_samples": len(window["thermal"]), "poses": poses,
                          "thermal_clock_motion_bracket_frame": window.get("thermal_view", {}).get("index"),
                          "thermal_clock_motion_alpha": window.get("thermal_alpha", 0.),
                          "landmark_depth_change_p95_m": float(np.percentile(np.abs(P[:, 2]-window["initial_points"][:, 2]), 95))})
    return fit.x[:6], {"solver_success": bool(fit.success), "termination": fit.message,
                      "evaluations": fit.nfev, "cost": float(fit.cost), "optimality": float(fit.optimality),
                      "parameter_count": len(initial), "residual_count": len(residual(initial)), "windows": summaries,
                      "seed_extrinsic": seed.x.tolist(), "annotation_sigma_px": sigma_px,
                      "gauge": "Each local window fixes its central RGB pose; estimated depth anchors fix scale.",
                      "priors": "depth sigma=max(0.03m,5%Z); RGB pixel sigma=1px; pose priors=0.02rad/0.05m; shared extrinsic priors=0.05rad/0.005m (CAD translation anchor); line samples share total line weight",
                      "limitations": "Local window BA with shared static extrinsics; not a globally loop-closed trajectory or surveyed calibration."}


def evaluate_horizontal_observations(dataset, calibration, refined_pose, observations_path):
    observations = json.loads(Path(observations_path).read_text())
    records = []
    for index in sorted({o["index"] for o in observations["observations"]}):
        sample = load_frame(dataset, index)
        points, _ = depth_cloud(sample["depth"], calibration["rgb"], 1)
        selected, uv = thermal_correspondence(points, calibration["transform"], calibration["thermal"],
                                              calibration["rgb"]["frame"], verified=True)
        tree = cKDTree(uv)
        for observation in observations["observations"]:
            if observation["index"] != index:
                continue
            query = [observation["projected_rgb_x"], observation["row_y"]]
            distance, nearest = tree.query(query)
            if distance > 2:
                raise ValueError("Held-out observation lacks a nearby valid depth sample")
            point = points[selected[nearest]]
            before = project_pose(point[None], np.r_[np.zeros(3), calibration["transform"]["t"]], calibration["thermal"]["K"])[0]
            after = project_pose(point[None], refined_pose, calibration["thermal"]["K"])[0]
            records.append({**observation, "source_depth_m": float(point[2]),
                            "before_x": float(before[0]), "after_x": float(after[0]),
                            "before_abs_horizontal_px": float(abs(before[0]-observation["thermal_x"])),
                            "after_abs_horizontal_px": float(abs(after[0]-observation["thermal_x"]))})
    return records


def refine_scene(root, profile_path, annotations_path, heldout_path, output, report_path):
    cv2.setNumThreads(4)
    cv2.setRNGSeed(42)
    dataset = open_dataset(root)
    calibration = load_fusion_calibration(root, profile_path)
    annotations = json.loads(Path(annotations_path).read_text())
    heldout = json.loads(Path(heldout_path).read_text())
    if set(annotations["training_frames"]) & {o["index"] for o in heldout["observations"]}:
        raise ValueError("Training and held-out observations overlap")
    windows = collect_scene_windows(dataset, calibration, annotations)
    pose, report = fit_scene_bundle(windows, calibration, annotations["annotation_sigma_px"])
    observations = evaluate_horizontal_observations(dataset, calibration, pose, heldout_path)
    before = np.array([o["before_abs_horizontal_px"] for o in observations])
    after = np.array([o["after_abs_horizontal_px"] for o in observations])
    accepted = bool(report["solver_success"] and after.mean() < before.mean() and after.max() < before.max())
    record = {"method": "windowed RGB-D / thermal structural-line bundle adjustment",
              "accepted_for_sequence": accepted, "base_profile_sha256": calibration["evidence"]["profile_sha256"],
              "dataset_revision": dataset["manifest"]["revision"], "sequence": dataset["manifest"]["sequence"],
              "annotations_sha256": hashlib.sha256(Path(annotations_path).read_bytes()).hexdigest(),
              "training_frames": annotations["training_frames"], "heldout_frames": annotations["heldout_frames"],
              "rgb_to_thermal_R": cv2.Rodrigues(pose[:3])[0].tolist(), "rgb_to_thermal_t_m": pose[3:].tolist(),
              "rvec_rad": pose[:3].tolist(), "heldout_observations": observations,
              "heldout_horizontal_before_mean_max_px": [float(before.mean()), float(before.max())],
              "heldout_horizontal_after_mean_max_px": [float(after.mean()), float(after.max())],
              "status": "scene_refined_approximate", "physical_accuracy_certified": False,
              "validation_scope": "Six manually marked horizontal edges across three held-out frames; approximately 4px annotation uncertainty; not full 2D accuracy or cross-sequence validation."}
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps({**record, "bundle_adjustment": report}, indent=2) + "\n")
    if accepted:
        Path(output).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ["accepted_for_sequence", "rvec_rad", "rgb_to_thermal_t_m", "heldout_horizontal_before_mean_max_px", "heldout_horizontal_after_mean_max_px"]}, indent=2))
    if not accepted:
        raise ValueError("Refinement did not pass the declared held-out improvement gate")
