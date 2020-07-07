import sys
import os
import json
import numpy as np

INDIR = sys.argv[1]
OUTDIR = sys.argv[2]


def _make_pose_file(dir, name, offset=0):
    """ only for testing purposes """
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


os.makedirs(OUTDIR, exist_ok=True)
_make_pose_file(OUTDIR, "alphapose-results")
