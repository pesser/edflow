import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope, arg_scope
import math

from matplotlib import pyplot as plt

import tensorflow.contrib.distributions as tfd


# TODO: write tests


def model_arg_scope(**kwargs):
    """Create new counter and apply arg scope to all arg scoped nn
    operations."""
    counters = {}
    return arg_scope(
        [conv2d, deconv2d, residual_block, dense, activate], counters=counters, **kwargs
    )


def make_model(name, template, **kwargs):
    """Create model with fixed kwargs."""
    run = lambda *args, **kw: template(
        *args, **dict((k, v) for kws in (kw, kwargs) for k, v in kws.items())
    )
    if tf.executing_eagerly():
        return tf.make_template(name, run)
    return tf.make_template(name, run, unique_name=name)


def int_shape(x):
    """ short for x.shape.as_list() """
    return x.shape.as_list()


def get_name(layer_name, counters):
    """ utlity for keeping track of layer names """
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + "_" + str(counters[layer_name])
    counters[layer_name] += 1
    return name


def apply_partwise(input_, func):
    """
    Applies function func on all parts separately.
    Parts are in channel 3.
    The input is reshaped to map the parts to the batch axis and then the function is applied
    Parameters
    ----------
    input_ : tensor
        [b, h, w, parts, features]
    func : callable
        a NN function to apply to each part individually
    Returns
    -------
        [b, out_h, out_w, parts, out_features]
    """

    b, h, w, parts, f = input_.shape.as_list()

    # transpose [b, h, w, part, features] --> [part, b, h, w, features]
    perm = [3, 0, 1, 2, 4]
    x = tf.transpose(input_, perm=perm)
    # reshape [part, b, h, w, features] --> [part * b, h, w, features]
    x = tf.reshape(x, (b * parts, h, w, f))

    y = func(x)

    _, h_out, w_out, c_out = y.shape.as_list()
    # reshape [part * b, h_out, w_out, c_out] --> [part, b, h_out, w_out, c_out]
    out = tf.reshape(y, (parts, b, h_out, w_out, c_out))
    # transpose back [part, b, h_out, w_out, c_out] --> [b, h_out, w_out, part, c_out]
    inv_perm = [1, 2, 3, 0, 4]
    out = tf.transpose(out, perm=inv_perm)
    return out


@add_arg_scope
def partwise_conv2d(
    x,
    num_filters,
    filter_size=[3, 3],
    stride=[1, 1],
    pad="SAME",
    init_scale=1.0,
    counters={},
    init=False,
    initdist="uniform",
    **kwargs
):
    """
    input: [b, h, w, parts, features]
    Each part (channel 3) has is its own bias and scale
    Uses 3D convolution internally to prevent tf.transpose

    Examples
    --------

    import tensorflow as tf
    tf.enable_eager_execution()

    from pylab import *
    from skimage import data
    import numpy as np
    import math
    im = data.astronaut()
    im = im.astype(np.float32) / 255
    H, W, D = im.shape

    b = 1
    parts = 5
    out_features = 1
    features = tf.reshape(im, (b, H, W, 1, D))
    features = tf.concat([features] * parts, axis=3)

    out = partwise_conv2d(features, out_features, init=False, part_wise=True, initdist="debug")

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    a = np.hstack([np.squeeze(out[..., p, :]) for p in range(parts)])
    ax.imshow(a, cmap=plt.cm.gray) # this should render the astronaut in 5 different shades of gray

    """
    num_filters = int(num_filters)
    name = get_name("conv2d", counters)
    with tf.variable_scope(name):
        in_channels = x.shape.as_list()[4]
        in_parts = x.shape.as_list()[3]
        fan_in = in_channels * filter_size[0] * filter_size[1]
        stdv = math.sqrt(1.0 / fan_in)
        part_stdv = math.sqrt(1.0 / in_parts)
        if initdist == "uniform":
            V_initializer = tf.random_uniform_initializer(minval=-stdv, maxval=stdv)
            b_initializer = tf.random_uniform_initializer(minval=-stdv, maxval=stdv)
            v_part_initializer = tf.random_uniform_initializer(
                minval=-part_stdv, maxval=part_stdv
            )
            b_part_initializer = tf.random_uniform_initializer(
                minval=-part_stdv, maxval=part_stdv
            )
        elif initdist == "normal":
            V_initializer = tf.random_normal_initializer(stddev=stdv)
            b_initializer = tf.random_normal_initializer(stddev=stdv)
            v_part_initializer = tf.random_normal_initializer(stddev=part_stdv)
            b_part_initializer = tf.random_normal_initializer(stddev=part_stdv)
        elif initdist == "debug":
            pass
        else:
            raise ValueError(initdist)
        if not initdist == "debug":
            V = tf.get_variable(
                "V",
                filter_size + [1, in_channels, num_filters],
                initializer=V_initializer,
                dtype=tf.float32,
            )
            b = tf.get_variable(
                "b",
                [1, 1, 1, 1, num_filters],
                initializer=b_initializer,
                dtype=tf.float32,
            )
            V_part = tf.get_variable(
                "V_part",
                [1, 1, 1, in_parts, 1],
                initializer=v_part_initializer,
                dtype=tf.float32,
            )
            b_part = tf.get_variable(
                "b_part",
                [1, 1, 1, in_parts, 1],
                initializer=b_part_initializer,
                dtype=tf.float32,
            )
        else:
            V = (
                tf.reshape([1.0, 2.0, 1.0], (3, 1, 1, 1, 1))
                / 4
                * tf.reshape([1.0, 2.0, 1.0], (1, 3, 1, 1, 1))
                / 4
                * tf.reshape([1.0, 1.0, 0.0], (1, 1, 1, 3, 1))
                / 2
            )
            b = tf.zeros([1, 1, 1, 1, num_filters], dtype=tf.float32)
            V_part = tf.reshape(
                tf.cast(tf.linspace(0.0, 1.0, in_parts), dtype=tf.float32),
                [1, 1, 1, in_parts, 1],
            )
            b_part = tf.zeros([1, 1, 1, in_parts, 1], dtype=tf.float32)
        x = tf.nn.conv3d(x, V, strides=[1] + stride + [1] + [1], padding="SAME")
        x *= V_part
        x += b + b_part
        return x


def _conv2d(
    x,
    num_filters,
    filter_size=[3, 3],
    stride=[1, 1],
    pad="SAME",
    init_scale=1.0,
    counters={},
    init=False,
    **kwargs
):
    num_filters = int(num_filters)
    strides = [1] + stride + [1]
    name = get_name("conv2d", counters)
    initdist = "uniform"
    with tf.variable_scope(name):
        in_channels = int(x.get_shape()[-1])
        fan_in = in_channels * filter_size[0] * filter_size[1]
        stdv = math.sqrt(1.0 / fan_in)
        if initdist == "uniform":
            V_initializer = tf.random_uniform_initializer(minval=-stdv, maxval=stdv)
            b_initializer = tf.random_uniform_initializer(minval=-stdv, maxval=stdv)
        elif initdist == "normal":
            V_initializer = tf.random_normal_initializer(stddev=stdv)
            b_initializer = tf.random_normal_initializer(stddev=stdv)
        else:
            raise ValueError(initdist)
        V = tf.get_variable(
            "V",
            filter_size + [in_channels, num_filters],
            initializer=V_initializer,
            dtype=tf.float32,
        )
        b = tf.get_variable(
            "b", [num_filters], initializer=b_initializer, dtype=tf.float32
        )
        if init:
            tmp = tf.nn.conv2d(x, V, [1] + stride + [1], pad) + tf.reshape(
                b, [1, 1, 1, num_filters]
            )
            mean, var = tf.nn.moments(tmp, [0, 1, 2])
            scaler = 1.0 / tf.sqrt(var + 1e-6)
            V = tf.assign(V, V * scaler)
            b = tf.assign(b, -mean * scaler)
        x = tf.nn.conv2d(x, V, [1] + stride + [1], pad) + tf.reshape(
            b, [1, 1, 1, num_filters]
        )
        return x


@add_arg_scope
def conv2d(
    x,
    num_filters,
    filter_size=[3, 3],
    stride=[1, 1],
    pad="SAME",
    init_scale=1.0,
    counters={},
    init=False,
    part_wise=False,
    coords=False,
    **kwargs
):
    """
        A 2D convolution.
    Parameters
    ----------
    x: tensor
        input tensor [N, H, W, C]
    num_filters: int
        number of feature maps
    filter_size: list of `ints`
        filter size in x, y
    stride: list of `ints`
        stride in x, y
    pad: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
        option 'SAME'
    init_scale
    counters
    init
    part_wise: bool
        if True, the input has to be [N, H, W, Parts, C]. The convolution will get an additional scale and bias per part
    coords: bool
        if True, will use coordConv (2018ACS_liuIntriguingFailingConvolutionalNeuralNetworks)
    kwargs

    Returns tensor
        convolved input
    -------

    """
    if coords:
        x = add_coordinates(x)
    if part_wise:
        out = partwise_conv2d(
            x,
            num_filters,
            filter_size=filter_size,
            stride=stride,
            pad=pad,
            init_scale=init_scale,
            counters=counters,
            init=init,
            **kwargs
        )
    else:
        out = _conv2d(
            x,
            num_filters,
            filter_size=filter_size,
            stride=stride,
            pad=pad,
            init_scale=init_scale,
            counters=counters,
            init=init,
            **kwargs
        )

    return out


@add_arg_scope
def dense(x, num_units, init_scale=1.0, counters={}, init=False, **kwargs):
    """ fully connected layer """
    name = get_name("dense", counters)
    initdist = "uniform"
    with tf.variable_scope(name):
        in_channels = int(x.get_shape()[-1])
        fan_in = in_channels
        stdv = math.sqrt(1.0 / fan_in)
        if initdist == "uniform":
            V_initializer = tf.random_uniform_initializer(minval=-stdv, maxval=stdv)
            b_initializer = tf.random_uniform_initializer(minval=-stdv, maxval=stdv)
        elif initdist == "normal":
            V_initializer = tf.random_normal_initializer(stddev=stdv)
            b_initializer = tf.random_normal_initializer(stddev=stdv)
        else:
            raise ValueError(initdist)
        V = tf.get_variable(
            "V", [in_channels, num_units], initializer=V_initializer, dtype=tf.float32
        )
        b = tf.get_variable(
            "b", [num_units], initializer=b_initializer, dtype=tf.float32
        )
        if init:
            tmp = tf.matmul(x, V) + tf.reshape(b, [1, num_units])
            mean, var = tf.nn.moments(tmp, [0])
            scaler = 1.0 / tf.sqrt(var + 1e-6)
            V = tf.assign(V, V * scaler)
            b = tf.assign(b, -mean * scaler)
        x = tf.matmul(x, V) + tf.reshape(b, [1, num_units])
        return x


@add_arg_scope
def activate(x, activation, **kwargs):
    """
    Activation unit
    Parameters
    ----------
    x: tensor
        input tensor
    activation:
        A `string` from: `"elu", "relu", "leaky_relu", "softplus"`.
    kwargs

    Returns
    -------

    """
    if activation == None:
        return x
    elif activation == "elu":
        return tf.nn.elu(x)
    elif activation == "relu":
        return tf.nn.relu(x)
    elif activation == "leaky_relu":
        return tf.nn.leaky_relu(x)
    elif activation == "softplus":
        return tf.nn.softplus(x)
    else:
        raise NotImplemented(activation)


def nin(x, num_units):
    """ a network in network layer (1x1 CONV) """
    return conv2d(x, num_units, filter_size=[1, 1])


def downsample(x, num_units):
    """
    Downsampling by stride 2 convolution

    equivalent to
    x = conv2d(x, num_units, stride = [2, 2])

    Parameters
    ----------
    x: tensor
        input
    num_units:
        number of feature map in the output

    Returns
    -------
    """
    return conv2d(x, num_units, stride=[2, 2])


def upsample(x, num_units, method="subpixel"):
    """
        2D upsampling layer.
    Parameters
    ----------
    x: tensor
        input
    num_units:
        number of feature maps in the output
    method:
        upsampling method. A `string` from: `"conv_transposed", "nearest_neighbor", "linear", "subpixel"`
        Subpixel means that every upsampled pixel gets its own filter.

    Returns
        upsampled input
    -------
    """
    xs = x.shape.as_list()
    if method == "conv_transposed":
        return deconv2d(x, num_units, stride=[2, 2])
    elif method == "subpixel":
        x = conv2d(x, 4 * num_units)
        x = tf.depth_to_space(x, 2)
        return x
    elif method == "nearest_neighbor":
        bs, h, w, c = x.shape.as_list()
        x = tf.image.resize_images(
            x, [2 * h, 2 * w], tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        return x
    elif method == "linear":
        bs, h, w, c = xs[:4]
        x = tf.image.resize_images(x, [2 * h, 2 * w], tf.image.ResizeMethod.BILINEAR)
        return x
    else:
        raise NotImplemented(method)


@add_arg_scope
def partwise_deconv2d(
    x,
    num_filters,
    filter_size=[3, 3],
    stride=[1, 1],
    pad="SAME",
    init_scale=1.0,
    counters={},
    init=False,
    **kwargs
):
    """ transposed convolutional layer """
    num_filters = int(num_filters)
    name = get_name("deconv2d", counters)
    xs = x.shape.as_list()
    strides = [1] + stride + [1]
    in_parts = xs[3]
    part_stdv = math.sqrt(1.0 / in_parts)
    v_part_initializer = tf.random_uniform_initializer(
        minval=-part_stdv, maxval=part_stdv
    )
    b_part_initializer = tf.random_uniform_initializer(
        minval=-part_stdv, maxval=part_stdv
    )
    if pad == "SAME":
        target_shape = [
            xs[0] * in_parts,
            xs[1] * stride[0],
            xs[2] * stride[1],
            num_filters,
        ]
    else:
        target_shape = [
            xs[0] * in_parts,
            xs[1] * stride[0] + filter_size[0] - 1,
            xs[2] * stride[1] + filter_size[1] - 1,
            num_filters,
        ]
    with tf.variable_scope(name):
        V = tf.get_variable(
            "V",
            filter_size + [num_filters, xs[-1]],
            tf.float32,
            tf.random_normal_initializer(0, 0.05),
        )
        g = tf.get_variable(
            "g",
            [num_filters],
            dtype=tf.float32,
            initializer=tf.constant_initializer(1.0),
        )
        b = tf.get_variable(
            "b",
            [num_filters],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0),
        )
        V_part = tf.get_variable(
            "V_part", [in_parts], initializer=v_part_initializer, dtype=tf.float32
        )
        b_part = tf.get_variable(
            "b_part", [in_parts], initializer=b_part_initializer, dtype=tf.float32
        )
        V_norm = tf.nn.l2_normalize(V, [0, 1, 3])

        def part_conv_func(x_):
            x_ = tf.nn.conv2d_transpose(
                x_, V_norm, target_shape, [1] + stride + [1], pad
            )
            x_ = tf.reshape(g, [1, 1, 1, num_filters]) * x_ + tf.reshape(
                b, [1, 1, 1, num_filters]
            )
            return x_

        if init:
            mean, var = tf.nn.moments(x, [0, 1, 2])
            g = tf.assign(g, init_scale / tf.sqrt(var + 1e-10))
            b = tf.assign(b, -mean * g)
        x = apply_partwise(x, part_conv_func)
        x = x * tf.reshape(V_part, (1, 1, 1, in_parts, 1)) + tf.reshape(
            b_part, (1, 1, 1, in_parts, 1)
        )
        return x


@add_arg_scope
def _deconv2d(
    x,
    num_filters,
    filter_size=[3, 3],
    stride=[1, 1],
    pad="SAME",
    init_scale=1.0,
    counters={},
    init=False,
    **kwargs
):
    """ transposed convolutional layer """
    num_filters = int(num_filters)
    name = get_name("deconv2d", counters)
    xs = x.shape.as_list()
    strides = [1] + stride + [1]
    if pad == "SAME":
        target_shape = [xs[0], xs[1] * stride[0], xs[2] * stride[1], num_filters]
    else:
        target_shape = [
            xs[0],
            xs[1] * stride[0] + filter_size[0] - 1,
            xs[2] * stride[1] + filter_size[1] - 1,
            num_filters,
        ]
    with tf.variable_scope(name):
        V = tf.get_variable(
            "V",
            filter_size + [num_filters, xs[-1]],
            tf.float32,
            tf.random_normal_initializer(0, 0.05),
        )
        g = tf.get_variable(
            "g",
            [num_filters],
            dtype=tf.float32,
            initializer=tf.constant_initializer(1.0),
        )
        b = tf.get_variable(
            "b",
            [num_filters],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0),
        )

        V_norm = tf.nn.l2_normalize(V, [0, 1, 3])
        x = tf.nn.conv2d_transpose(x, V_norm, target_shape, [1] + stride + [1], pad)
        if init:
            mean, var = tf.nn.moments(x, [0, 1, 2])
            g = tf.assign(g, init_scale / tf.sqrt(var + 1e-10))
            b = tf.assign(b, -mean * g)
        x = tf.reshape(g, [1, 1, 1, num_filters]) * x + tf.reshape(
            b, [1, 1, 1, num_filters]
        )
        return x


@add_arg_scope
def deconv2d(
    x,
    num_filters,
    filter_size=[3, 3],
    stride=[1, 1],
    pad="SAME",
    init_scale=1.0,
    counters={},
    init=False,
    part_wise=False,
    coords=False,
    **kwargs
):
    """
        coords: if True, will use coordConv (2018ACS_liuIntriguingFailingConvolutionalNeuralNetworks)
    """
    if coords:
        x = add_coordinates(x)
    if part_wise:
        out = partwise_deconv2d(
            x,
            num_filters,
            filter_size=filter_size,
            stride=stride,
            pad=pad,
            init_scale=init_scale,
            counters=counters,
            init=init,
            **kwargs
        )
    else:
        out = _deconv2d(
            x,
            num_filters,
            filter_size=filter_size,
            stride=stride,
            pad=pad,
            init_scale=init_scale,
            counters=counters,
            init=init,
            **kwargs
        )
    return out


@add_arg_scope
def residual_block(x, skipin=None, conv=conv2d, init=False, dropout_p=0.0, **kwargs):
    """
    slight variation of original residual block.

    Parameters
    ----------
    x: tensor
        Incoming tensor
    skipin: tensor
        Incomming tensor from skip connection, if any
    conv: callable
        which convolution function to use for the resodual
    init
    dropout_p: float
        dropout probability, if any
    kwargs

    Returns
    -------
    output : x + residual



    Examples


    """
    xs = int_shape(x)
    num_filters = xs[-1]

    residual = x
    if skipin is not None:
        skipin = nin(activate(skipin), num_filters)
        residual = tf.concat([residual, skipin], axis=-1)
    residual = activate(residual)
    residual = tf.nn.dropout(residual, keep_prob=1.0 - dropout_p)
    residual = conv(residual, num_filters)

    return x + residual


def flatten(x):
    """
    returns a flat version of x --> [N, -1]
    Parameters
    ----------
    x: tensor


    Returns
    -------

    """
    _shape = x.shape.as_list()
    return tf.reshape(x, (_shape[0], -1))


def mask2rgb(mask):
    """
    Convert tensor with masks [N, H, W, C] to an RGB tensor [N, H, W, 3]
    using argmax over channels.
    Parameters
    ----------
    mask: ndarray
        an array of shape [N, H, W, C]
    Returns:
        RGB visualization in shape [N, H, W, 3]
    -------

    """
    n_parts = mask.shape.as_list()[3]
    maxmask = tf.argmax(mask, axis=3)
    hotmask = tf.one_hot(maxmask, depth=n_parts)
    hotmask = tf.expand_dims(hotmask, 4)
    colors = make_mask_colors(n_parts)
    colors = (colors - 0.5) * 2
    colors = tf.to_float(colors)
    colors = tf.expand_dims(colors, 0)
    colors = tf.expand_dims(colors, 0)
    colors = tf.expand_dims(colors, 0)
    vis_mask = hotmask * colors
    vis_mask = tf.reduce_sum(vis_mask, axis=3)
    return vis_mask


def np_one_hot(targets, n_classes):
    """
    numpy equivalent of tf.one_hot
    returns targets as one hot matrix

    Parameters
    ----------
    targets: ndarray
        array of target classes
    n_classes: int
        how many classes there are overall
    Returns: ndarray
        one-hot array with shape [n, n_classes]
    -------

    """
    res = np.eye(n_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [n_classes])


def np_to_float(x):
    """ cast x to float32 """
    return x.astype(np.float32)


def np_mask2rgb(mask):
    """
    numpy equivalent of @mask2rgb
    convert tensor with masks [N, H, W, C] to an RGB tensor [N, H, W, 3]
    using argmax over channels.
    Parameters
    ----------
    mask: ndarray
        an array of shape [N, H, W, C]
    Returns:
        RGB visualization in shape [N, H, W, 3]
    -------

    """
    n_parts = mask.shape[3]
    maxmask = np.argmax(mask, axis=3)
    hotmask = np_one_hot(maxmask, n_classes=n_parts)
    hotmask = np.expand_dims(hotmask, 4)
    colors = make_mask_colors(n_parts)
    colors = (colors - 0.5) * 2
    colors = np_to_float(colors)
    colors = np.expand_dims(colors, 0)
    colors = np.expand_dims(colors, 0)
    colors = np.expand_dims(colors, 0)
    vis_mask = hotmask * colors
    vis_mask = np.sum(vis_mask, axis=3)
    return vis_mask


def make_mask_colors(n_parts, cmap=plt.cm.inferno):
    """
    make a color array using the specified colormap for n_parts classes
    Parameters
    ----------
    n_parts: int
        how many classes there are in the mask
    cmap:
        matplotlib colormap handle

    Returns
    -------
    colors: ndarray
        an array with shape [n_parts, 3] representing colors in the range [0, 1].

    """
    colors = cmap(np.linspace(0, 1, n_parts), alpha=False, bytes=False)[:, :3]
    return colors


def hourglass_model(
    x,
    config,
    extra_resnets,
    n_out=3,
    activation="relu",
    upsample_method="subpixel",
    coords=False,
):
    """
    A U-net or hourglass style image-to-image model with skip-connections

    Parameters
    ----------
    x: tensor
        input tensor to unet
    config: list
        a list of ints specifying the number of feature maps on each scale of the unet in the downsampling path
        for the upsampling path, the list will be reversed
        For example [32, 64] will use 32 channels on scale 0 (without downsampling) and 64 channels on scale 1
        once downsampled).
    extra_resnets : int
        how many extra res blocks to use at the bottleneck
    n_out : int
        number of final output feature maps of the unet. 3 for RGB
    activation: str
        a string specifying the activation function to use. See @activate for options.
    upsample_method: list of str or str
        a str specifying the upsampling method or a list of str specifying the upsampling method for each scale individually.
        See @upsample for possible options.
    coords: True
        if coord conv should be used.

    Returns
    -------


    Examples
    --------

    tf.enable_eager_execution()
    x = tf.ones((1, 128, 128, 3))
    config = [32, 64]
    extra_resnets = 0
    upsample_method = "subpixel"
    activation = "leaky_relu"
    coords = False

    unet = make_model("unet", hourglass_model,
                      config=config, extra_resnets= extra_resnets, upsample_method=upsample_method, activation=activation)
    y = unet(x)

    # plotting the output should look random because we did not train anything
    im = np.concatenate([x, y], axis=1)
    plt.imshow(np.squeeze(im))

    """
    with model_arg_scope(activation=activation, coords=coords):
        hs = list()

        h = conv2d(x, config[0])
        h = residual_block(h)

        for nf in config[1:]:
            h = downsample(h, nf)
            h = residual_block(h)
            hs.append(h)

        for _ in range(extra_resnets):
            h = residual_block(h)

        for i, nf in enumerate(config[-2::-1]):
            h = residual_block(h, skipin=hs[-(i + 1)])
            h = upsample(h, nf, method=upsample_method)

        h = residual_block(h)
        h = conv2d(h, n_out)
        return h


def make_ema(init_value, value, decay=0.99):
    """
    apply exponential moving average to variable

    Parameters
    ----------
    init_value: float
        initial value for moving average variable
    value: variable
        tf variable to apply update ops on
    decay: float
        decay parameter

    Returns
    -------
    avg_value : variable with exponential moving average
    update_ema: tensorflow update operation for exponential moving average

    Examples
    --------

    # usage within edflow Trainer.make_loss_ops. Apply EMA to discriminator accuracy
    avg_acc, update_ema = make_ema(0.5, dis_accuracy, decay)
    self.update_ops.append(update_ema)
    self.log_ops["dis_acc"] = avg_acc


    """
    decay = tf.constant(decay, dtype=tf.float32)
    avg_value = tf.Variable(init_value, dtype=tf.float32, trainable=False)
    update_ema = tf.assign(
        avg_value, decay * avg_value + (1.0 - decay) * tf.cast(value, tf.float32)
    )
    return avg_value, update_ema


def add_coordinates(input_tensor, with_r=False):
    """
    Given an input_tensor, adds 2 channelw ith x and y coordinates to the feature maps.
    This was introduced in coordConv (2018ACS_liuIntriguingFailingConvolutionalNeuralNetworks).
    Parameters
    ----------
    input_tensor: tensor
        Tensor of shape [N, H, W, C]
    with_r : bool
        if True, euclidian radius will also be added as channel

    Returns
    ret : input_tensor concatenated with x and y coordinates and maybe euclidian distance.
    -------

    """
    assert len(input_tensor.shape.as_list()) == 4
    bs = tf.shape(input_tensor)[0]
    x_dim, y_dim = input_tensor.shape.as_list()[1:3]
    xx_ones = tf.ones([bs, x_dim], dtype=tf.int32)
    xx_ones = tf.expand_dims(xx_ones, -1)
    xx_range = tf.tile(tf.expand_dims(tf.range(y_dim), 0), [bs, 1])
    xx_range = tf.expand_dims(xx_range, 1)

    xx_channel = tf.matmul(xx_ones, xx_range)
    xx_channel = tf.expand_dims(xx_channel, -1)

    yy_ones = tf.ones([bs, y_dim], dtype=tf.int32)
    yy_ones = tf.expand_dims(yy_ones, 1)
    yy_range = tf.tile(tf.expand_dims(tf.range(x_dim), 0), [bs, 1])
    yy_range = tf.expand_dims(yy_range, -1)

    yy_channel = tf.matmul(yy_range, yy_ones)
    yy_channel = tf.expand_dims(yy_channel, -1)

    xx_channel = tf.cast(xx_channel, "float32") / max(1, x_dim - 1)
    yy_channel = tf.cast(yy_channel, "float32") / max(1, y_dim - 1)
    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1

    ret = tf.concat([input_tensor, xx_channel, yy_channel], axis=-1)
    if with_r:
        rr = tf.sqrt(tf.square(xx_channel) + tf.square(yy_channel))
        ret = tf.concat([ret, rr], axis=-1)
    return ret


def probs_to_mu_L(
    probs, scaling_factor, inv=True
):  # todo maybe exponential map induces to much certainty ! low values basically ignored and only high values count!
    """
        Calculate mean and covariance for each channel of probs
        tensor of keypoint probabilites [bn, h, w, n_kp]
        mean calculated on a grid of scale [-1, 1]
    Parameters
    ----------
    probs: tensor
        tensor of shape [b, h, w, k] where each channel along axis 3 is interpreted as an unnormalized probability density.
    scaling_factor : tensor
        tensor of shape [b, 1, 1, k] representing normalizing the normalizing constant of the density
    inv: bool
        if True, returns covariance matrix of density. Else returns inverse of covariance matrix aka precision matrix

    Returns
    -------
    mu : tensor
        tensor of shape [b, k, 2] representing partwise mean coordinates of x and y for each item in the batch
    L : tensor
        tensor of shape [b, k, 2, 2] representing partwise cholesky decomposition of covariance
         matrix for each item in the batch.

    Examples
    --------

    from matplotlib import pyplot as plt
    tf.enable_eager_execution()
    import numpy as np
    import tensorflow.contrib.distributions as tfd

    _means = [-0.5, 0, 0.5]
    means = tf.ones((3, 1, 2), dtype=tf.float32) * np.array(_means).reshape((3, 1, 1))
    means = tf.concat([means, means, means[::-1, ...]], axis=1)
    means = tf.reshape(means, (-1, 2))

    var_ = 0.1
    rho = 0.5
    cov = [[var_, rho * var_],
           [rho * var_, var_]]
    scale = tf.cholesky(cov)
    scale = tf.stack([scale] * 3, axis=0)
    scale = tf.stack([scale] * 3, axis=0)
    scale = tf.reshape(scale, (-1, 2, 2))

    mvn = tfd.MultivariateNormalTriL(
        loc=means,
        scale_tril=scale)

    h = 100
    w = 100
    y_t = tf.tile(tf.reshape(tf.linspace(-1., 1., h), [h, 1]), [1, w])
    x_t = tf.tile(tf.reshape(tf.linspace(-1., 1., w), [1, w]), [h, 1])
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

    """
    bn, h, w, nk = (
        probs.get_shape().as_list()
    )  # todo instead of calulating sequrity measure from amplitude one could alternativly calculate it by letting the network predict a extra paremeter also one could do
    y_t = tf.tile(tf.reshape(tf.linspace(-1.0, 1.0, h), [h, 1]), [1, w])
    x_t = tf.tile(tf.reshape(tf.linspace(-1.0, 1.0, w), [1, w]), [h, 1])
    y_t = tf.expand_dims(y_t, axis=-1)
    x_t = tf.expand_dims(x_t, axis=-1)
    meshgrid = tf.concat([y_t, x_t], axis=-1)

    mu = tf.einsum("ijl,aijk->akl", meshgrid, probs)
    mu_out_prod = tf.einsum(
        "akm,akn->akmn", mu, mu
    )  # todo incosisntent ordereing of mu! compare with cross_V2

    mesh_out_prod = tf.einsum(
        "ijm,ijn->ijmn", meshgrid, meshgrid
    )  # todo efficient (expand_dims)
    stddev = tf.einsum("ijmn,aijk->akmn", mesh_out_prod, probs) - mu_out_prod

    a_sq = stddev[:, :, 0, 0]
    a_b = stddev[:, :, 0, 1]
    b_sq_add_c_sq = stddev[:, :, 1, 1]
    eps = 1e-12  # todo clean magic

    a = tf.sqrt(
        a_sq + eps
    )  # Σ = L L^T Prec = Σ^-1  = L^T^-1 * L^-1  ->looking for L^-1 but first L = [[a, 0], [b, c]
    b = a_b / (a + eps)
    c = tf.sqrt(b_sq_add_c_sq - b ** 2 + eps)
    z = tf.zeros_like(a)

    if inv:
        det = tf.expand_dims(tf.expand_dims(a * c, axis=-1), axis=-1)
        row_1 = tf.expand_dims(
            tf.concat(
                [tf.expand_dims(c, axis=-1), tf.expand_dims(z, axis=-1)], axis=-1
            ),
            axis=-2,
        )
        row_2 = tf.expand_dims(
            tf.concat(
                [tf.expand_dims(-b, axis=-1), tf.expand_dims(a, axis=-1)], axis=-1
            ),
            axis=-2,
        )

        L_inv = (
            scaling_factor / (det + eps) * tf.concat([row_1, row_2], axis=-2)
        )  # L^⁻1 = 1/(ac)* [[c, 0], [-b, a]
        return mu, L_inv
    else:
        row_1 = tf.expand_dims(
            tf.concat(
                [tf.expand_dims(a, axis=-1), tf.expand_dims(z, axis=-1)], axis=-1
            ),
            axis=-2,
        )
        row_2 = tf.expand_dims(
            tf.concat(
                [tf.expand_dims(b, axis=-1), tf.expand_dims(c, axis=-1)], axis=-1
            ),
            axis=-2,
        )

        L = scaling_factor * tf.concat([row_1, row_2], axis=-2)  # just L
        return mu, L


def tf_hm(h, w, mu, L, order="exp"):
    """
        Returns Gaussian densitiy function based on μ and L for each batch index and part
        L is the cholesky decomposition of the covariance matrix : Σ = L L^T
    Parameters
    ----------
    h : int
        heigh ot output map
    w : int
        width of output map
    mu : tensor
        mean of gaussian part and batch item. Shape [b, p, 2]. Mean in range [-1, 1] with respect to height and width
    L : tensor
        cholesky decomposition of covariance matrix for each batch item and part. Shape [b, p, 2, 2]
    order:

    Returns
    -------
    density : tensor
        gaussian blob for each part and batch idx. Shape [b, h, w, p]

    Examples
    --------

    from matplotlib import pyplot as plt
    tf.enable_eager_execution()
    import numpy as np
    import tensorflow as tf
    import tensorflow.contrib.distributions as tfd

    # create Target Blobs
    _means = [-0.5, 0, 0.5]
    means = tf.ones((3, 1, 2), dtype=tf.float32) * np.array(_means).reshape((3, 1, 1))
    means = tf.concat([means, means, means[::-1, ...]], axis=1)
    means = tf.reshape(means, (-1, 2))

    var_ = 0.1
    rho = 0.5
    cov = [[var_, rho * var_],
           [rho * var_, var_]]
    scale = tf.cholesky(cov)
    scale = tf.stack([scale] * 3, axis=0)
    scale = tf.stack([scale] * 3, axis=0)
    scale = tf.reshape(scale, (-1, 2, 2))

    mvn = tfd.MultivariateNormalTriL(
        loc=means,
        scale_tril=scale)

    h = 100
    w = 100
    y_t = tf.tile(tf.reshape(tf.linspace(-1., 1., h), [h, 1]), [1, w])
    x_t = tf.tile(tf.reshape(tf.linspace(-1., 1., w), [1, w]), [h, 1])
    y_t = tf.expand_dims(y_t, axis=-1)
    x_t = tf.expand_dims(x_t, axis=-1)
    meshgrid = tf.concat([y_t, x_t], axis=-1)
    meshgrid = tf.expand_dims(meshgrid, 0)
    meshgrid = tf.expand_dims(meshgrid, 3)  # 1, h, w, 1, 2

    blob = mvn.prob(meshgrid)
    blob = tf.reshape(blob, (100, 100, 3, 3))
    blob = tf.transpose(blob, perm=[2, 0, 1, 3])

    # Estimate mean and L
    norm_const = np.sum(blob, axis=(1, 2), keepdims=True)
    mu, L = nn.probs_to_mu_L(blob / norm_const, 1, inv=False)

    bn, h, w, nk = blob.get_shape().as_list()

    # Estimate blob based on mu and L
    estimated_blob = nn.tf_hm(h, w, mu, L)

    # plot
    fig, ax = plt.subplots(2, 3, figsize=(9, 6))
    for b in range(len(_means)):
        ax[0, b].imshow(np.squeeze(blob[b, ...]))
        ax[0, b].set_title("target_blobs")
        ax[0, b].set_axis_off()

    for b in range(len(_means)):
        ax[1, b].imshow(np.squeeze(estimated_blob[b, ...]))
        ax[1, b].set_title("estimated_blobs")
        ax[1, b].set_axis_off()

    """

    assert len(mu.get_shape().as_list()) == 3
    assert len(L.get_shape().as_list()) == 4
    assert mu.get_shape().as_list()[-1] == 2
    assert L.get_shape().as_list()[-1] == 2
    assert L.get_shape().as_list()[-2] == 2

    b, p, _ = mu.get_shape().as_list()
    mu = tf.reshape(mu, (b * p, 2))
    L = tf.reshape(L, (b * p, 2, 2))

    mvn = tfd.MultivariateNormalTriL(loc=mu, scale_tril=L)
    y_t = tf.tile(tf.reshape(tf.linspace(-1.0, 1.0, h), [h, 1]), [1, w])
    x_t = tf.tile(tf.reshape(tf.linspace(-1.0, 1.0, w), [1, w]), [h, 1])
    y_t = tf.expand_dims(y_t, axis=-1)
    x_t = tf.expand_dims(x_t, axis=-1)
    meshgrid = tf.concat([y_t, x_t], axis=-1)
    meshgrid = tf.expand_dims(meshgrid, 0)
    meshgrid = tf.expand_dims(meshgrid, 3)  # 1, h, w, 1, 2

    probs = mvn.prob(meshgrid)
    probs = tf.reshape(probs, (h, w, b, p))
    probs = tf.transpose(probs, perm=[2, 0, 1, 3])  # move part axis to the back
    return probs


if __name__ == "__main__":
    tf.enable_eager_execution()
    x = tf.ones((1, 128, 128, 3))
    config = [32, 64]
    extra_resnets = 0
    upsample_method = "subpixel"
    activation = "leaky_relu"
    coords = False

    unet = make_model(
        "unet",
        hourglass_model,
        config=config,
        extra_resnets=extra_resnets,
        upsample_method=upsample_method,
        activation=activation,
    )
    y = unet(x)

    im = np.concatenate([x, y], axis=1)
    plt.imshow(np.squeeze(im))
