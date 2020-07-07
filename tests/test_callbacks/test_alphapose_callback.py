import pytest
import numpy as np
import os
import json
from edflow.callbacks import alphapose_callback


def _make_pose_file(dir, name, offset=0):
    keypoints = np.arange(64).reshape((32, 2)) - offset
    scores = np.ones((32, 1), dtype=np.float16)
    keypoint_data = np.concatenate([keypoints, scores], axis=1).ravel()
    fname = os.path.join(dir, name) + ".json"
    image_id = "img_01.png"

    pose_data = [
        {
            "keypoints": list(keypoint_data),
            "score": 3.0,
            "bbox": (0, 0, 128, 128),
            "category_id": 1,
            "image_id": image_id,
            "idx": [0.0],
        },
    ]
    with open(fname, "w") as f:
        json.dump(pose_data, f)
    return fname


def test_read_pose_file(tmpdir):
    fname_1 = _make_pose_file(tmpdir, "alphapose_1")
    from edflow.callbacks import alphapose_callback

    pose_data = alphapose_callback.read_posefile(fname_1)
    keypoints = pose_data[0]["keypoints"]
    assert len(keypoints) == 96


def test_pck_from_posefiles(tmpdir):
    fname_1 = _make_pose_file(tmpdir, "alphapose_1")
    fname_2 = _make_pose_file(tmpdir, "alphapose_2")

    mean_pck = alphapose_callback.pck_from_posefiles(
        fname_1, fname_2, distance_threshold=10
    )
    assert mean_pck == 1.0

    fname_3 = _make_pose_file(tmpdir, "alphapose_3", offset=10)

    mean_pck = alphapose_callback.pck_from_posefiles(
        fname_1, fname_3, distance_threshold=10
    )
    assert mean_pck == 0.0
