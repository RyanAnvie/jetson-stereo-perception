import numpy as np

from multimodal_perception.camera import transform_points


def backproject(pixels, depth_m, camera, units="m"):
    """Return N x 3 optical-frame points: x right, y down, z forward, meters."""
    if units != "m":
        raise ValueError("Depth must be in meters")
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    depth_m = np.asarray(depth_m, dtype=float).reshape(-1)
    if len(pixels) != len(depth_m):
        raise ValueError("One depth value is required per pixel")
    K = camera["K"]
    valid = np.isfinite(pixels).all(axis=1) & np.isfinite(depth_m) & (depth_m > 0)
    points = np.full((len(pixels), 3), np.nan)
    points[valid, 0] = (pixels[valid, 0] - K[0, 2]) * depth_m[valid] / K[0, 0]
    points[valid, 1] = (pixels[valid, 1] - K[1, 2]) * depth_m[valid] / K[1, 1]
    points[valid, 2] = depth_m[valid]
    return points, valid


def project(points, camera):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    pixels = np.full((len(points), 2), np.nan)
    K = camera["K"]
    pixels[valid, 0] = K[0, 0] * points[valid, 0] / points[valid, 2] + K[0, 2]
    pixels[valid, 1] = K[1, 1] * points[valid, 1] / points[valid, 2] + K[1, 2]
    valid &= (pixels[:, 0] >= 0) & (pixels[:, 0] < camera["width"])
    valid &= (pixels[:, 1] >= 0) & (pixels[:, 1] < camera["height"])
    return pixels, valid


def depth_cloud(depth, camera, stride=6):
    if depth.shape != (camera["height"], camera["width"]):
        raise ValueError("Depth dimensions do not match calibration")
    if stride < 1:
        raise ValueError("Point stride must be positive")
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    pixels = np.column_stack((u.ravel(), v.ravel()))
    points, valid = backproject(pixels, depth[v, u].ravel(), camera)
    return points[valid], pixels[valid]


def thermal_correspondence(points, transform, camera, source_frame, verified=False):
    """Nearest-Z per thermal pixel; only usable after image-domain verification."""
    if not verified:
        raise ValueError("Thermal image projection is unverified; association is blocked")
    if camera["frame"] != transform["target"]:
        raise ValueError("Thermal camera and transform frames differ")
    transformed = transform_points(points, transform, source_frame)
    pixels, valid = project(transformed, camera)
    indices = np.flatnonzero(valid)
    integer_pixels = np.floor(pixels[indices]).astype(int)
    keys = integer_pixels[:, 1] * camera["width"] + integer_pixels[:, 0]
    order = np.argsort(transformed[indices, 2], kind="stable")
    _, nearest = np.unique(keys[order], return_index=True)
    keep = order[nearest]
    return indices[keep], integer_pixels[keep]


def clip_box(box, width, height):
    box = np.asarray(box, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("Bounding box must contain four finite xyxy values")
    x1 = int(np.clip(np.floor(box[0]), 0, width))
    y1 = int(np.clip(np.floor(box[1]), 0, height))
    x2 = int(np.clip(np.ceil(box[2]), 0, width))
    y2 = int(np.clip(np.ceil(box[3]), 0, height))
    return x1, y1, x2, y2


def locate_detection(box, depth, camera, min_points=20):
    x1, y1, x2, y2 = clip_box(box, camera["width"], camera["height"])
    if depth.shape != (camera["height"], camera["width"]):
        raise ValueError("Depth dimensions do not match calibration")
    crop = depth[y1:y2, x1:x2]
    valid = np.isfinite(crop) & (crop > 0)
    count = int(valid.sum())
    result = {"point_m": None, "frame": camera["frame"], "valid_depth_points": count,
              "valid_depth_fraction": count / max(crop.size, 1),
              "status": "insufficient_depth", "thermal_status": "calibration_unverified"}
    if count < min_points:
        return result
    v, u = np.nonzero(valid)
    pixels = np.column_stack((u + x1, v + y1))
    points, _ = backproject(pixels, crop[valid], camera)
    # Componentwise median over the full box; background contamination is possible.
    point = np.median(points, axis=0)
    result.update(point_m=point.tolist(), status="ok",
                  depth_mad_m=float(np.median(np.abs(points[:, 2] - np.median(points[:, 2])))))
    return result
