import cv2
import numpy as np
import pytest


@pytest.mark.parametrize("thermal_alpha", [0., .6])
def test_scene_bundle_recovers_known_rotation_and_composed_thermal_poses(thermal_alpha):
    pytest.importorskip("scipy")
    from multimodal_perception.scene_calibration import fit_scene_bundle, project_pose
    K = np.array([[350., 0, 320], [0, 350, 256], [0, 0, 1]])
    rng = np.random.default_rng(5)
    points = rng.uniform([-2., -1., 3.], [2., 1., 8.], (40, 3))
    true_extrinsic = np.array([.015, -.018, .005, 0., -.03166579, 0.])
    neighbor = np.array([.002, .004, -.003, .04, 0, .01])
    moving = neighbor*thermal_alpha
    thermal_points = points @ cv2.Rodrigues(moving[:3])[0].T+moving[3:]
    heat_uv = project_pose(thermal_points, true_extrinsic, K)
    thermal = []
    for i, uv in enumerate(heat_uv):
        # Independent horizontal and vertical structural constraints.
        for normal in [np.array([1., 0.]), np.array([0., 1.])]:
            thermal.append({"point": i, "normal": normal, "offset": -normal @ uv, "sigma": 1.})
    window = {"anchor": 10, "initial_points": points, "thermal": thermal,
              "views": [{"index": 10, "pose": np.zeros(6), "point_indices": np.arange(40),
                         "uv": project_pose(points, np.zeros(6), K)},
                        {"index": 12, "pose": neighbor, "point_indices": np.arange(40),
                         "uv": project_pose(points, neighbor, K)}]}
    calibration = {"rgb": {"K": K}, "thermal": {"K": K}, "transform": {"t": true_extrinsic[3:]}}
    if thermal_alpha:
        window.update(thermal_view=window["views"][1], thermal_alpha=thermal_alpha)
    pose, report = fit_scene_bundle([window], calibration, 1.)
    assert report["solver_success"]
    np.testing.assert_allclose(pose[:3], true_extrinsic[:3], atol=.0005)
    np.testing.assert_allclose(pose[3:], true_extrinsic[3:], atol=.001)
    target = report["windows"][0]["poses"][1]
    rgb_pose = np.array(target["rgb_from_window_rvec_t"])
    expected_rotation = cv2.Rodrigues(pose[:3])[0] @ cv2.Rodrigues(rgb_pose[:3])[0]
    np.testing.assert_allclose(target["thermal_from_window_R"], expected_rotation)
    assert target["rgb_reprojection_p50_p95_px"][1] < .1
