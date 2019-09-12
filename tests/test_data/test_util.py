import os

import numpy as np

from edflow.data.util import plot_datum


def test_plot_datum():
    test_image = np.ones((128, 128, 3), dtype=int)
    test_heatmap = np.zeros((128, 128, 25), dtype=int)
    test_keypoints = np.random.randint(0, 128, (25, 2))

    test_example = {
        "image": test_image,
        "heatmap": test_heatmap,
        "keypoints": test_keypoints,
    }
    plot_datum(test_example, "test_plot.png")
    assert os.path.exists("test_plot.png")
    os.remove("test_plot.png")
