import numpy as np
import pytest

from multimodal_perception.camera import pinhole_camera, resize_camera, rigid_transform, inverse_transform, transform_points
from multimodal_perception.geometry import backproject, project, thermal_correspondence, locate_detection, clip_box


def test_backprojection_reprojection():
    camera = pinhole_camera([[100, 0, 50], [0, 100, 40], [0, 0, 1]], 100, 80, "rgb")
    pixels = np.array([[50, 40], [20, 30], [90, 70]])
    points, valid = backproject(pixels, [2, 3, 4], camera)
    np.testing.assert_allclose(points[0], [0, 0, 2])
    projected, visible = project(points, camera)
    np.testing.assert_allclose(projected, pixels)
    assert valid.all() and visible.all()


def test_nonidentity_inverse_and_units():
    rotation = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    transform = rigid_transform(rotation, [1, 2, 3], "a", "b")
    points = np.array([[1, 0, 2], [3, 2, 1]])
    output = transform_points(points, transform, "a")
    np.testing.assert_allclose(output[0], [1, 3, 5])
    np.testing.assert_allclose(transform_points(output, inverse_transform(transform), "b"), points)
    with pytest.raises(ValueError):
        transform_points(points, transform, "wrong")
    with pytest.raises(ValueError):
        rigid_transform(rotation, [1, 2, 3], "a", "b", "mm")
    with pytest.raises(ValueError):
        rigid_transform(np.ones((3, 3)), [0, 0, 0], "a", "b")


def test_invalid_depth_and_fov():
    camera = pinhole_camera(np.eye(3), 10, 10, "rgb")
    points, valid = backproject([[0, 0]] * 5, [1, 0, -1, np.nan, np.inf], camera)
    assert valid.tolist() == [True, False, False, False, False]
    assert np.isnan(points[1:]).all()
    _, valid = project([[1, 1, 1], [0, 0, -1], [10, 1, 1], [np.nan, 1, 1]], camera)
    assert valid.tolist() == [True, False, False, False]
    with pytest.raises(ValueError):
        backproject([[0, 0]], [1000], camera, "mm")


def test_resize_uses_pixel_centers():
    camera = pinhole_camera([[100, 0, 49.5], [0, 100, 39.5], [0, 0, 1]], 100, 80, "rgb")
    scaled = resize_camera(camera, 50, 40)
    np.testing.assert_allclose(scaled["K"], [[50, 0, 24.5], [0, 50, 19.5], [0, 0, 1]])
    restored = resize_camera(scaled, 100, 80)
    np.testing.assert_allclose(restored["K"], camera["K"])


def test_zbuffer_and_calibration_gate():
    camera = pinhole_camera(np.eye(3), 10, 10, "thermal")
    transform = rigid_transform(np.eye(3), [0, 0, 0], "rgb", "thermal")
    points = np.array([[2, 2, 2], [1, 1, 1], [100, 100, 1]])
    with pytest.raises(ValueError, match="unverified"):
        thermal_correspondence(points, transform, camera, "rgb")
    indices, pixels = thermal_correspondence(points, transform, camera, "rgb", verified=True)
    assert indices.tolist() == [1]
    assert pixels.tolist() == [[1, 1]]


def test_bbox_clipping_and_insufficient_depth():
    camera = pinhole_camera([[10, 0, 5], [0, 10, 5], [0, 0, 1]], 10, 10, "rgb")
    depth = np.full((10, 10), 2, dtype=float)
    depth[:2] = np.nan
    assert clip_box([-5, -2, 15, 12], 10, 10) == (0, 0, 10, 10)
    result = locate_detection([-5, -2, 15, 12], depth, camera)
    assert result["status"] == "ok" and result["valid_depth_points"] == 80
    assert result["point_m"][2] == 2
    assert locate_detection([2, 2, 2, 2], depth, camera)["status"] == "insufficient_depth"
