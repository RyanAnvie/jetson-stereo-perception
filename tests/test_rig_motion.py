import cv2
import numpy as np
import pytest

from multimodal_perception.camera import pinhole_camera, rigid_transform
from multimodal_perception.rig_motion import interpolate_rgb_motion, exposure_transform, estimate_rgb_motion, calibration_at_exposure


def test_exposure_composition_preserves_rigid_extrinsic_and_direction():
    R = cv2.Rodrigues(np.array([.02, -.01, .03]))[0]
    fixed = rigid_transform(R, [.01, -.03, .002], "rgb", "thermal")
    before = fixed["R"].copy(), fixed["t"].copy()
    calibration = {"transform": fixed}
    motion = {"accepted": True, "dt_s": .1, "rvec_rad": [.03, .01, -.02], "translation_m": [-.03, 0, .01]}
    moving_R, moving_t = interpolate_rgb_motion(motion, .07)
    result = exposure_transform(calibration, motion, .07)
    point = np.array([.2, .3, 3.])
    np.testing.assert_allclose(result["R"]@point+result["t"], R@(moving_R@point+moving_t)+fixed["t"])
    np.testing.assert_array_equal(fixed["R"], before[0]); np.testing.assert_array_equal(fixed["t"], before[1])
    start = exposure_transform(calibration, motion, 0)
    np.testing.assert_allclose(start["R"], R); np.testing.assert_allclose(start["t"], fixed["t"])
    for outside in [-.001, .101]:
        with pytest.raises(ValueError, match="bracket"):
            interpolate_rgb_motion(motion, outside)
    with pytest.raises(ValueError, match="quality"):
        interpolate_rgb_motion({**motion, "accepted": False}, .05)


def test_rgb_motion_recovers_planar_scene_translation_and_rejects_blank():
    rng = np.random.default_rng(12)
    gray = rng.integers(0, 256, (540, 960), np.uint8)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    K = np.array([[400., 0, 480], [0, 400., 270], [0, 0, 1]])
    camera = pinhole_camera(K, 960, 540, "rgb")
    shift = np.float32([[1, 0, -4], [0, 1, 2]])
    source = {"index": 0, "rgb_timestamp_s": 0., "rgb": rgb, "depth": np.full((540, 960), 4., np.float32)}
    target = {"index": 1, "rgb_timestamp_s": .1, "rgb": cv2.warpAffine(rgb, shift, (960, 540))}
    motion = estimate_rgb_motion(source, target, camera)
    assert motion["accepted"]
    np.testing.assert_allclose(motion["translation_m"], [-.04, .02, 0], atol=.003)
    np.testing.assert_allclose(motion["rvec_rad"], [0, 0, 0], atol=.001)
    blank = np.zeros_like(rgb)
    rejected = estimate_rgb_motion({**source, "rgb": blank}, {**target, "rgb": blank}, camera)
    assert not rejected["accepted"]


def test_missing_motion_is_explicit_and_never_changes_rig_extrinsics():
    calibration = {"evidence": {"refinement": {"thermal_clock_correction_s": .07}}}
    sample = {"rgb_timestamp_s": 1., "thermal_timestamp_s": 1.002}
    unchanged, record = calibration_at_exposure(sample, calibration, True)
    assert unchanged is calibration and not record["compensated"]
    assert record["status"] == "missing_rgb_bracket" and record["fixed_extrinsics"]
    assert abs(record["effective_thermal_timestamp_s"]-1.072) < 1e-12
