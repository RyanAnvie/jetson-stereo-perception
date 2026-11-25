import numpy as np
import pytest

from multimodal_perception.dataset import frame_files, match_timestamps
from multimodal_perception.detection import preprocess, nonmaximum_suppression, COCO_CLASSES


def test_timestamp_pairing_checks_sidecars():
    target = np.array([100.0, 100.1, 100.2])
    actual = np.array([100.01, 100.15, -1])
    errors = np.array([0.01, 0.05, -1])
    valid, _ = match_timestamps(target, actual, errors, 0.032)
    assert valid.tolist() == [True, False, False]
    with pytest.raises(ValueError, match="contradict"):
        match_timestamps(target, actual, [0.02, 0.05, -1], 0.032)
    with pytest.raises(ValueError, match="length"):
        match_timestamps(target, actual[:2], errors, 0.032)
    with pytest.raises(ValueError, match="increasing"):
        match_timestamps(target[::-1], actual, errors, 0.032)


def test_numeric_frame_indices(tmp_path):
    for name in ["10.png", "2.png", "0.png"]:
        (tmp_path / name).touch()
    assert sorted(frame_files(tmp_path, ".png")) == [0, 2, 10]


def test_yolox_input_is_bgr_not_normalized():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:] = [10, 20, 30]
    tensor, ratio = preprocess(image)
    assert ratio == 2.08 and tensor.shape == (1, 3, 416, 416)
    assert tensor.dtype == np.float32
    np.testing.assert_allclose(tensor[0, :, 0, 0], [10, 20, 30])
    np.testing.assert_allclose(tensor[0, :, -1, 0], [114, 114, 114])
    assert len(COCO_CLASSES) == 80 and COCO_CLASSES[0] == "person"


def test_nms_removes_overlap_not_distant_boxes():
    boxes = np.array([[0., 0., 10., 10.], [1., 1., 9., 9.], [20., 20., 30., 30.]])
    scores = np.array([0.9, 0.8, 0.7])
    assert nonmaximum_suppression(boxes, scores, 0.45) == [0, 2]
