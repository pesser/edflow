from skimage import img_as_float
from skimage.measure import compare_ssim
import numpy as np


def ssim_metric(batch1, batch2):
    """Compute the sctructural similarity score."""
    S = []
    # Assumption is made that there is a batch size
    assert len(batch1.shape) == 4
    for a, b in zip(batch1, batch2):
        s = compare_ssim(a, b, multichannel=True)
        S += [s]
    return np.array(S)


def l2_metric(batch1, batch2):
    """Pixelwise l2 distance mean."""
    diff = batch1 - batch2
    diff = np.reshape(diff, [diff.shape[0], -1])
    return np.linalg.norm(diff, axis=1)
