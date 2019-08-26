import pytest
import tensorflow as tf

from edflow.nn import tf_nn as nn


def test_int_shape():
    tf.enable_eager_execution()
    a = tf.ones((1, 2, 3, 4))
    a_shape = nn.int_shape(a)
    assert type(a_shape) is list


def test_partwise_conv2d():
    from matplotlib import pyplot as plt
    from skimage import data
    import numpy as np

    im = data.astronaut()
    im = im.astype(np.float32) / 255
    H, W, D = im.shape

    b = 1
    parts = 5
    out_features = 1
    features = tf.reshape(im, (b, H, W, 1, D))
    features = tf.concat([features] * parts, axis=3)

    out = nn.partwise_conv2d(
        features, out_features, init=False, part_wise=True, initdist="debug"
    )
    # fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    # a = np.hstack([np.squeeze(out[..., p, :]) for p in range(parts)])
    # fig.tight_layout()
    # ax.imshow(a, cmap=plt.cm.gray) # this should render the astronaut in 5 different shades of gray

    coefficients = np.linspace(0, 1, parts)
    assert np.allclose(out[..., 0, :], np.zeros_like(out[..., 0, :]))
    assert np.allclose(out[..., -1, :] * coefficients[-2], out[..., -2, :])


def test_conv2d():
    from skimage import data
    import numpy as np

    im = data.astronaut()
    im = im.astype(np.float32) / 255
    H, W, D = im.shape

    b = 1
    x = tf.reshape(im, (b, H, W, D))

    out = nn.conv2d(x, 128)

    assert out.shape == (1, 512, 512, 128)


def test_dense():
    x = tf.ones((1, 100), dtype=tf.float32)
    out = nn.dense(x, 512)
    assert out.shape == (1, 512)


def test_upsample():
    from skimage import data
    import numpy as np

    im = data.astronaut()
    im = im.astype(np.float32) / 255
    H, W, D = im.shape

    b = 1
    x = tf.reshape(im, (b, H, W, D))

    out = nn.upsample(x, 3, method="conv_transposed")
    assert out.shape == (1, 1024, 1024, 3)
    out = nn.upsample(x, 3, method="subpixel")
    assert out.shape == (1, 1024, 1024, 3)
    out = nn.upsample(x, 3, method="nearest_neighbor")
    assert out.shape == (1, 1024, 1024, 3)
    out = nn.upsample(x, 3, method="linear")
    assert out.shape == (1, 1024, 1024, 3)


def test_mask2rgb():
    import numpy as np

    m = tf.random_normal((1, 512, 512, 10))
    mask = tf.nn.softmax(m, axis=-1)

    rgb_mask = nn.mask2rgb(mask)
    assert rgb_mask.shape == (1, 512, 512, 3)

    mask_np = np.array(mask)
    rgb_mask_numpy = nn.np_mask2rgb(mask_np)
    assert np.allclose(rgb_mask, rgb_mask_numpy)


def test_blobs():
    from matplotlib import pyplot as plt

    import matplotlib

    # fix from https://youtrack.jetbrains.com/issue/PY-29684?_ga=2.3673904.1244701922.1566840489-1923801516.1548153235
    matplotlib.use("module://backend_interagg")

    tf.enable_eager_execution()
    import numpy as np
    import tensorflow.contrib.distributions as tfd

    _means = [-0.5, 0, 0.5]
    means = tf.ones((3, 1, 2), dtype=tf.float32) * np.array(_means).reshape((3, 1, 1))
    means = tf.concat([means, means, means[::-1, ...]], axis=1)
    means = tf.reshape(means, (-1, 2))

    var_ = 0.1
    rho = 0.5
    cov = [[var_, rho * var_], [rho * var_, var_]]
    scale = tf.cholesky(cov)
    scale = tf.stack([scale] * 3, axis=0)
    scale = tf.stack([scale] * 3, axis=0)
    scale = tf.reshape(scale, (-1, 2, 2))

    mvn = tfd.MultivariateNormalTriL(loc=means, scale_tril=scale)

    h = 100
    w = 100
    y_t = tf.tile(tf.reshape(tf.linspace(-1.0, 1.0, h), [h, 1]), [1, w])
    x_t = tf.tile(tf.reshape(tf.linspace(-1.0, 1.0, w), [1, w]), [h, 1])
    y_t = tf.expand_dims(y_t, axis=-1)
    x_t = tf.expand_dims(x_t, axis=-1)
    meshgrid = tf.concat([y_t, x_t], axis=-1)
    meshgrid = tf.expand_dims(meshgrid, 0)
    meshgrid = tf.expand_dims(meshgrid, 3)  # 1, h, w, 1, 2

    blob = mvn.prob(meshgrid)
    blob = tf.reshape(blob, (100, 100, 3, 3))
    blob = tf.transpose(blob, perm=[2, 0, 1, 3])

    norm_const = np.sum(blob, axis=(1, 2), keepdims=True)
    mu, L = nn.probs_to_mu_L(blob / norm_const, 1, inv=False)

    bn, h, w, nk = blob.get_shape().as_list()
    estimated_blob = nn.tf_hm(h, w, mu, L)

    fig, ax = plt.subplots(2, 3, figsize=(9, 6))
    for b in range(len(_means)):
        ax[0, b].imshow(np.squeeze(blob[b, ...]))
        ax[0, b].set_title("target_blobs")
        ax[0, b].set_axis_off()

    for b in range(len(_means)):
        ax[1, b].imshow(np.squeeze(estimated_blob[b, ...]))
        ax[1, b].set_title("estimated_blobs")
        ax[1, b].set_axis_off()


class Test_MumfordSha(object):
    def test_forward_difference_kernel(self):
        import matplotlib.pyplot as plt

        import matplotlib

        # fix from https://youtrack.jetbrains.com/issue/PY-29684?_ga=2.3673904.1244701922.1566840489-1923801516.1548153235
        matplotlib.use("module://backend_interagg")

        tf.enable_eager_execution()
        import numpy as np

        x = np.zeros((1, 100, 100, 1), dtype=np.float32)
        x[:, 30:70, 30:70, :] = 1.0

        y = nn.tf_grad(tf.convert_to_tensor(x))

        fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        ax[0].imshow(np.squeeze(x))
        ax[0].set_axis_off()

        ax[1].imshow(np.squeeze(y[..., 0]))
        ax[1].set_axis_off()

        ax[2].imshow(np.squeeze(y[..., 1]))
        ax[2].set_axis_off()
        fig.tight_layout()
        plt.show()

        gradient_x = y[..., 0]
        gradient_y = y[..., 0]
        assert np.allclose(gradient_x, gradient_y[::-1, ::-1])
