"""KITTI raw synced/rectified images, calibration, scans and tracklets."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from multimodal_perception.camera import pinhole_camera, rigid_transform, transform_points
from multimodal_perception.geometry import project


def read_calibration_values(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        name, content = line.split(":", 1)
        if name == "calib_time":
            continue
        values[name] = np.array([float(value) for value in content.split()])
    return values


def load_kitti_calibration(directory):
    directory = Path(directory)
    values = read_calibration_values(directory / "calib_cam_to_cam.txt")
    velo = read_calibration_values(directory / "calib_velo_to_cam.txt")
    P_left = values["P_rect_02"].reshape(3, 4)
    P_right = values["P_rect_03"].reshape(3, 4)
    width, height = values["S_rect_02"].astype(int)
    if not np.array_equal(values["S_rect_02"], values["S_rect_03"]):
        raise ValueError("Stereo image sizes differ")
    camera = pinhole_camera(P_left[:, :3], width, height, "kitti_cam2_rect")
    pinhole_camera(P_right[:, :3], width, height, "kitti_cam3_rect")
    # P_i = K_i [I | t_i] maps rectified cam0 points into camera i pixels.
    # Retain all three translation components, including the small y/z offsets.
    t_left = np.linalg.solve(P_left[:, :3], P_left[:, 3])
    t_right = np.linalg.solve(P_right[:, :3], P_right[:, 3])
    R_rect = values["R_rect_00"].reshape(3, 3)
    R_velo = velo["R"].reshape(3, 3)
    t_velo = velo["T"]
    rotation = R_rect @ R_velo
    translation = R_rect @ t_velo + t_left
    transform = rigid_transform(rotation, translation, "kitti_velodyne", camera["frame"])
    baseline = float(t_left[0] - t_right[0])
    if baseline <= 0:
        raise ValueError("Expected right camera to the right of left camera")
    return {"camera": camera, "P_left": P_left, "P_right": P_right,
            "t_left": t_left, "t_right": t_right, "baseline_x_m": baseline,
            "T_cam2_from_velo": transform}


def read_kitti_timestamps(path):
    lines = Path(path).read_text().splitlines()
    dates = np.array(lines, dtype="datetime64[ns]")
    if np.isnat(dates).any():
        raise ValueError(f"Missing timestamp: {path}")
    timestamps = dates.astype(np.int64)
    if not len(timestamps) or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"Timestamps must increase strictly: {path}")
    # The source does not declare a timezone. Use integer differences only.
    return timestamps


def read_tracklets(path):
    root = ET.parse(path).getroot()
    container = root.find("tracklets")
    if container is None:
        raise ValueError("Missing tracklets XML element")
    frames = {}
    items = container.findall("item")
    if len(items) != int(container.findtext("count")):
        raise ValueError("Tracklet count mismatch")
    for track_id, item in enumerate(items):
        dimensions = np.array([float(item.findtext(key)) for key in ["h", "w", "l"]])
        if not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
            raise ValueError("Invalid tracklet dimensions")
        first = int(item.findtext("first_frame"))
        poses = item.find("poses")
        pose_items = poses.findall("item")
        if len(pose_items) != int(poses.findtext("count")):
            raise ValueError("Tracklet pose count mismatch")
        for offset, pose in enumerate(pose_items):
            translation = np.array([float(pose.findtext(key)) for key in ["tx", "ty", "tz"]])
            rx, ry, rz = [float(pose.findtext(key)) for key in ["rx", "ry", "rz"]]
            if not np.isfinite(translation).all() or not np.isfinite([rx, ry, rz]).all():
                raise ValueError("Nonfinite tracklet pose")
            # Official raw tracklet demo uses only yaw. Reject a different model.
            if abs(rx) > 1e-8 or abs(ry) > 1e-8:
                raise ValueError("Nonzero tracklet roll/pitch needs an explicit convention")
            rotation = np.array([[np.cos(rz), -np.sin(rz), 0],
                                 [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
            frames.setdefault(first+offset, []).append({
                "track_id": track_id, "class": item.findtext("objectType"),
                "dimensions_hwl_m": dimensions, "translation_velo_m": translation,
                "R_velo_from_object": rotation, "occlusion": int(pose.findtext("occlusion")),
                "truncation": int(pose.findtext("truncation"))})
    return frames, len(items)


def project_tracklet(tracklet, calibration):
    h, w, length = tracklet["dimensions_hwl_m"]
    corners = np.array([[length/2, w/2, 0], [length/2, -w/2, 0], [-length/2, -w/2, 0], [-length/2, w/2, 0],
                        [length/2, w/2, h], [length/2, -w/2, h], [-length/2, -w/2, h], [-length/2, w/2, h]])
    velo = (tracklet["R_velo_from_object"] @ corners.T).T + tracklet["translation_velo_m"]
    points = transform_points(velo, calibration["T_cam2_from_velo"], "kitti_velodyne")
    if not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
        return None
    camera = calibration["camera"]
    uv, _ = project(points, camera)
    bbox = np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, camera["width"])
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, camera["height"])
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return {**tracklet, "bbox_xyxy": bbox, "corners_uv": uv}


def points_in_tracklet(points_velo, tracklet):
    relative = points_velo - tracklet["translation_velo_m"]
    local = (tracklet["R_velo_from_object"].T @ relative.T).T
    h, w, length = tracklet["dimensions_hwl_m"]
    inside = np.isfinite(local).all(axis=1)
    inside &= (np.abs(local[:, 0]) <= length/2) & (np.abs(local[:, 1]) <= w/2)
    inside &= (local[:, 2] >= 0) & (local[:, 2] <= h)
    return inside


def open_kitti(config):
    root = Path(config["dataset_root"])
    directory = root / config["date"] / config["sequence"]
    calibration = load_kitti_calibration(root / config["date"])
    streams = {}
    for name, suffix in [("image_02", ".png"), ("image_03", ".png"), ("velodyne_points", ".bin")]:
        times = read_kitti_timestamps(directory / name / "timestamps.txt")
        files = {}
        for path in (directory / name / "data").glob("*"+suffix):
            index = int(path.stem)
            if index in files:
                raise ValueError("Duplicate frame index")
            files[index] = path
        if sorted(files) != list(range(len(times))):
            raise ValueError(f"Missing or extra frames in {name}")
        streams[name] = {"files": files, "timestamps_ns": times}
    count = len(streams["image_02"]["files"])
    if any(len(stream["files"]) != count for stream in streams.values()):
        raise ValueError("Synchronized stream lengths differ")
    times_left = streams["image_02"]["timestamps_ns"]
    stereo_delta_s = (streams["image_03"]["timestamps_ns"] - times_left) / 1e9
    valid = np.abs(stereo_delta_s) <= config["max_stereo_time_error_s"]
    tracklets, tracklet_count = read_tracklets(directory / "tracklet_labels.xml")
    if any(index < 0 or index >= count for index in tracklets):
        raise ValueError("Tracklet pose outside the image sequence")
    return {"root": root, "directory": directory, "streams": streams, "calibration": calibration,
            "valid": valid, "tracklets": tracklets, "tracklet_count": tracklet_count,
            "stereo_delta_s": stereo_delta_s,
            "manifest": json.loads((root / "sample_manifest.json").read_text())}


def load_kitti_frame(dataset, index, include_reference=True):
    streams = dataset["streams"]
    left = cv2.imread(str(streams["image_02"]["files"][index]), cv2.IMREAD_COLOR)
    right = cv2.imread(str(streams["image_03"]["files"][index]), cv2.IMREAD_COLOR)
    camera = dataset["calibration"]["camera"]
    expected = (camera["height"], camera["width"], 3)
    if left is None or right is None or left.shape != expected or right.shape != expected:
        raise ValueError(f"Image decode/calibration shape mismatch: {index}")
    sample = {"index": index, "left": left, "right": right,
              "timestamp_ns": int(streams["image_02"]["timestamps_ns"][index]),
              "right_timestamp_ns": int(streams["image_03"]["timestamps_ns"][index])}
    if include_reference:
        scan = np.fromfile(streams["velodyne_points"]["files"][index], dtype="<f4")
        if scan.size % 4:
            raise ValueError(f"Invalid LiDAR point record length: {index}")
        sample["lidar_xyzi"] = scan.reshape(-1, 4)
        sample["tracklets"] = dataset["tracklets"].get(index, [])
    return sample
