import numpy as np
import pytest

from multimodal_perception.camera import pinhole_camera, rigid_transform
from multimodal_perception.kitti import read_kitti_timestamps, read_tracklets, points_in_tracklet
from multimodal_perception.stereo import disparity_to_depth, project_lidar, create_stereo_matcher, estimate_stereo_depth
from multimodal_perception.kitti_evaluation import error_metrics, match_detections
from multimodal_perception.kitti_pipeline import replay_position


def test_disparity_depth_with_nonzero_fourth_column():
    K = np.array([[100., 0, 20], [0, 100, 15], [0, 0, 1]])
    t_left = np.array([0.06, 0.01, 0.003])
    t_right = np.array([-0.48, 0.02, 0.004])
    P_left = np.column_stack((K, K @ t_left))
    P_right = np.column_stack((K, K @ t_right))
    # Construct a point on each left pixel ray, then project through the right P.
    u = np.arange(40)
    depth_expected = 8.0
    x_cam2 = (u-K[0, 2])*depth_expected/K[0, 0]
    x_cam0 = x_cam2-t_left[0]
    z_cam0 = depth_expected-t_left[2]
    u_right = (K[0, 0]*(x_cam0+t_right[0])+K[0, 2]*(z_cam0+t_right[2]))/(z_cam0+t_right[2])
    disparity = (u-u_right)[None]
    depth = disparity_to_depth(disparity, P_left, P_right)
    np.testing.assert_allclose(depth, depth_expected, atol=1e-6)
    disparity[0, :3] = [0, -1, np.nan]
    assert np.isnan(disparity_to_depth(disparity, P_left, P_right)[0, :3]).all()


def test_timestamp_nanosecond_precision(tmp_path):
    path = tmp_path / "times.txt"
    path.write_text("2011-09-26 13:02:25.123456789\n2011-09-26 13:02:25.223456790\n")
    times = read_kitti_timestamps(path)
    assert times[1]-times[0] == 100000001
    path.write_text("2011-09-26 13:02:25.123456789\n2011-09-26 13:02:25.123456789\n")
    with pytest.raises(ValueError, match="increase"):
        read_kitti_timestamps(path)
    path.write_text("NaT\n")
    with pytest.raises(ValueError, match="Missing"):
        read_kitti_timestamps(path)


def test_lidar_zbuffer_preserves_scan_identity():
    camera = pinhole_camera(np.eye(3), 10, 10, "kitti_cam2_rect")
    transform = rigid_transform(np.eye(3), [0, 0, 0], "kitti_velodyne", camera["frame"])
    scan = np.array([[2, 2, 2, .1], [1, 1, 1, .8], [0, 0, -1, .4], [100, 0, 1, .2]])
    original = scan.copy()
    result = project_lidar(scan, {"camera": camera, "T_cam2_from_velo": transform})
    assert result["scan_indices"].tolist() == [1]
    np.testing.assert_equal(scan, original)


def test_tracklet_bottom_center_and_frame_offset(tmp_path):
    path = tmp_path / "tracklets.xml"
    path.write_text('''<boost_serialization><tracklets><count>1</count><item>
      <objectType>Car</objectType><h>2</h><w>2</w><l>4</l><first_frame>5</first_frame>
      <poses><count>1</count><item><tx>10</tx><ty>0</ty><tz>1</tz>
      <rx>0</rx><ry>0</ry><rz>1.5707963267948966</rz><occlusion>0</occlusion><truncation>0</truncation>
      </item></poses></item></tracklets></boost_serialization>''')
    frames, count = read_tracklets(path)
    assert count == 1 and list(frames) == [5]
    points = np.array([[10, 0, 2], [10, 1.5, 2], [11.5, 0, 2], [10, 0, 0.5], [10, 0, 3.1]])
    assert points_in_tracklet(points, frames[5][0]).tolist() == [True, True, False, False, False]


def test_matching_is_class_aware_and_one_to_one():
    detections = [{"class": "car", "bbox_xyxy": [0, 0, 10, 10]},
                  {"class": "car", "bbox_xyxy": [1, 1, 9, 9]},
                  {"class": "person", "bbox_xyxy": [0, 0, 10, 10]}]
    labels = [{"class": "Car", "bbox_xyxy": [0, 0, 10, 10]}]
    matches = match_detections(detections, labels, {"Car": "car"}, 0.5)
    assert matches == [{"detection_index": 0, "label_index": 0, "iou": 1.0}]


def test_error_metrics_keep_large_errors():
    result = error_metrics([-1, 1, 100], [10, 10, 10])
    assert result["count"] == 3 and result["mae_m"] == 34
    assert result["median_abs_error_m"] == 1
    assert error_metrics([], [])["mae_m"] is None
    with pytest.raises(ValueError):
        error_metrics([np.nan], [10])


def test_sgbm_on_known_shift():
    rng = np.random.default_rng(5)
    gray = rng.integers(0, 256, (96, 256), dtype=np.uint8)
    right_gray = np.zeros_like(gray)
    right_gray[:, :-8] = gray[:, 8:]
    left = np.repeat(gray[:, :, None], 3, axis=2)
    right = np.repeat(right_gray[:, :, None], 3, axis=2)
    parameters = {"block_size": 5, "num_disparities": 32, "min_disparity": 0,
                  "uniqueness_ratio": 10, "speckle_window_size": 0, "speckle_range": 2, "disp12_max_diff": 1}
    K = np.array([[100., 0, 128], [0, 100, 48], [0, 0, 1]])
    calibration = {"P_left": np.column_stack((K, [0, 0, 0])), "P_right": np.column_stack((K, [-50, 0, 0]))}
    depth, disparity = estimate_stereo_depth(left, right, create_stereo_matcher(parameters), calibration)
    central = depth[10:-10, 50:-20]
    assert np.isfinite(central).mean() > 0.95
    assert abs(np.nanmedian(central)-6.25) < 0.05
    assert abs(np.median(disparity[10:-10, 50:-20])-8) < 0.05


def test_replay_overload_drops_only_expired_positions():
    offsets = np.array([0., .1, .2, .3, .4])
    assert replay_position(offsets, 1, .35, True) == 3
    assert replay_position(offsets, 1, .35, False) == 1
    assert replay_position(offsets, 2, .15, True) == 2
    assert replay_position(offsets, 1, 2., True) == 4
    assert replay_position(offsets, 1, .2, True) == 2
