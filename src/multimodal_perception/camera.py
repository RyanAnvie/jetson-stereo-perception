import json
from pathlib import Path

import numpy as np
import yaml


def pinhole_camera(K, width, height, frame):
    K = np.asarray(K, dtype=float).reshape(3, 3)
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Invalid camera intrinsics")
    if not np.allclose(K[2], [0, 0, 1]) or K[0, 1] != 0 or K[1, 0] != 0:
        raise ValueError("Only zero-skew rectified pinhole intrinsics are supported")
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions")
    return {"K": K, "width": int(width), "height": int(height), "frame": frame,
            "domain": "rectified_pinhole", "units": "m"}


def resize_camera(camera, width, height):
    """OpenCV pixel-center resize convention, including the half-pixel offset."""
    sx = width / camera["width"]
    sy = height / camera["height"]
    K = camera["K"].copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return pinhole_camera(K, width, height, camera["frame"])


def rigid_transform(rotation, translation, source, target, units="m"):
    R = np.asarray(rotation, dtype=float).reshape(3, 3)
    t = np.asarray(translation, dtype=float).reshape(3)
    if units != "m":
        raise ValueError("Transforms must explicitly use meters")
    if not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError("Nonfinite transform")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(R), 1, atol=1e-7):
        raise ValueError("Rotation must be orthogonal with determinant +1")
    return {"R": R, "t": t, "source": source, "target": target, "units": units}


def inverse_transform(transform):
    R = transform["R"].T
    t = -R @ transform["t"]
    return rigid_transform(R, t, transform["target"], transform["source"])


def transform_points(points, transform, source, units="m"):
    if source != transform["source"] or units != transform["units"]:
        raise ValueError("Point frame or units do not match the transform")
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    return (transform["R"] @ points.T).T + transform["t"].reshape(1, 3)


def load_calibration(root):
    root = Path(root)
    rgb = yaml.safe_load((root / "released_zed_rectified_960x540.yaml").read_text())
    width, height = rgb["released_resolution"]
    if (width, height) != (960, 540):
        raise ValueError("Expected the released 960x540 RGB calibration")
    camera = pinhole_camera(rgb["rgb_left_rectified"]["K"], width, height, "zed_left_camera_optical_frame")
    tf = json.loads((root / "tf_static.json").read_text())
    entry = next(item for item in tf["transforms"] if item["parent"] == camera["frame"] and item["child"] == "thermal_left/optical_frame")
    q = np.asarray(entry["quaternion_xyzw"], dtype=float)
    if not np.isclose(np.linalg.norm(q), 1, atol=1e-6):
        raise ValueError("Quaternion must have unit length")
    x, y, z, w = q / np.linalg.norm(q)
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    # configs/handheld.yaml defines the stored TF as child pose in parent.
    parent_from_child = rigid_transform(R, entry["translation_xyz_m"], entry["child"], entry["parent"])
    thermal_from_rgb = inverse_transform(parent_from_child)
    return camera, thermal_from_rgb
