import json

import numpy as np
import pytest

from multimodal_perception.camera import pinhole_camera, rigid_transform
from multimodal_perception.thermal_fusion import rectified_thermal_camera, load_fusion_calibration, fuse_thermal
from multimodal_perception.geometry import thermal_correspondence


def test_producer_rectification_is_not_raw_projection():
    raw = {"image_width": 640, "image_height": 512, "distortion_model": "equidistant",
           "camera_matrix": {"data": [421.23237248, 0, 317.55165969, 0, 420.80872096, 255.54588954, 0, 0, 1]},
           "distortion_coefficients": {"data": [-0.11053445, 0.20481263, -0.57743943, 0.45121438]}}
    settings = {"size": [640, 512], "R": np.eye(3).tolist(), "balance": 0., "fov_scale": 1.,
                "K_expected": [[350.0586333355143, 0, 316.2060443676416],
                               [0, 349.7065643071332, 255.40731015684818], [0, 0, 1]]}
    camera = rectified_thermal_camera(raw, settings, "thermal")
    assert camera["domain"] == "rectified_pinhole"
    assert abs(camera["K"][0, 0] - raw["camera_matrix"]["data"][0]) > 70
    settings["balance"] = 1.
    with pytest.raises(ValueError, match="audited profile"):
        rectified_thermal_camera(raw, settings, "thermal")


def test_profile_blocks_unreviewed_release_and_changed_calibration(tmp_path):
    profile = {"dataset_revision": "audited", "validated_sequence": "sequence",
               "asset_sha256": {"calibration.yaml": "wrong-hash"}}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    manifest = tmp_path / "sample_manifest.json"
    manifest.write_text(json.dumps({"revision": "unknown", "sequence": "sequence"}))
    with pytest.raises(ValueError, match="revision"):
        load_fusion_calibration(tmp_path, path)
    manifest.write_text(json.dumps({"revision": "audited", "sequence": "other"}))
    with pytest.raises(ValueError, match="Sequence"):
        load_fusion_calibration(tmp_path, path)
    manifest.write_text(json.dumps({"revision": "audited", "sequence": "sequence"}))
    (tmp_path / "calibration.yaml").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        load_fusion_calibration(tmp_path, path)


def fusion_fixture():
    rgb = pinhole_camera(np.eye(3), 10, 10, "rgb")
    thermal = pinhole_camera([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]], 10, 10, "thermal")
    calibration = {"rgb": rgb, "thermal": thermal,
                   "transform": rigid_transform(np.eye(3), [0, 0, 0], "rgb", "thermal")}
    depth = np.full((10, 10), np.nan)
    depth[1, 1], depth[2, 2] = 2., 1.
    sample = {"timestamp_s": 1., "rgb_timestamp_s": 1., "thermal_timestamp_s": 1.002,
              "valid": True, "ffc": False, "depth": depth, "thermal": np.full((10, 10), 123, np.uint8)}
    detections = [{"bbox_xyxy": [1, 1, 2, 2], "position": {}}, {"bbox_xyxy": [2, 2, 3, 3], "position": {}}]
    return sample, detections, calibration


def test_global_occlusion_before_box_association_and_intensity_units():
    sample, detections, calibration = fusion_fixture()
    result = fuse_thermal(sample, detections, calibration, stride=1, min_points=1)
    assert result["rgb_pixels"].tolist() == [[2, 2]]
    assert detections[0]["thermal"]["support_points"] == 0
    assert detections[0]["thermal"]["status"] == "insufficient_support"
    association = detections[1]["thermal"]
    assert association["status"] == "associated_nominal"
    assert association["bbox_xyxy"] == [0, 0, 1, 1]
    assert association["intensity_p10_p50_p90"] == [123, 123, 123]
    assert association["units"] == "processed_intensity_u8"


def test_invalid_groups_and_bad_depth_never_create_thermal_associations():
    for key, value in [("valid", False), ("ffc", True), ("thermal_timestamp_s", 1.1), ("rgb_timestamp_s", np.nan)]:
        sample, detections, calibration = fusion_fixture()
        sample[key] = value
        result = fuse_thermal(sample, detections, calibration, stride=1, min_points=1)
        assert len(result["points"]) == 0
        assert all(d["thermal"]["status"] == "invalid_group" for d in detections)
    sample, detections, calibration = fusion_fixture()
    sample["depth"][:] = np.nan
    sample["depth"][1, 1], sample["depth"][2, 2] = 0., -1.
    result = fuse_thermal(sample, detections, calibration, stride=1, min_points=1)
    assert len(result["points"]) == 0
    assert all(d["thermal"]["support_points"] == 0 for d in detections)


def test_linear_zbuffer_matches_stable_sort_with_collisions_and_ties():
    rng = np.random.default_rng(42)
    uv = rng.uniform(0, 10, (1000, 2))
    z = rng.integers(1, 5, 1000)
    points = np.column_stack((uv * z[:, None], z))
    camera = pinhole_camera(np.eye(3), 10, 10, "thermal")
    transform = rigid_transform(np.eye(3), [0, 0, 0], "rgb", "thermal")
    keys = np.floor(uv[:, 1]).astype(int)*10 + np.floor(uv[:, 0]).astype(int)
    order = np.argsort(z, kind="stable")
    _, first = np.unique(keys[order], return_index=True)
    expected = order[first]
    selected, pixels = thermal_correspondence(points, transform, camera, "rgb", verified=True)
    np.testing.assert_array_equal(selected, expected)
    np.testing.assert_array_equal(pixels, np.floor(uv[expected]).astype(int))
