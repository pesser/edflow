from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import tensorflow as tf

import math


def tf_batch_to_canvas(X, cols: int = None):
    """reshape a batch of images into a grid canvas to form a single image.

    Parameters
    ----------
    X : Tensor
        Batch of images to format. [N, H, W, C]-shaped
    cols : int

    cols: int :
         (Default value = None)

    Returns
    -------
    image_grid : Tensor
        Tensor representing the image grid. [1, HH, WW, C]-shaped

    Examples
    --------

    x = np.ones((9, 100, 100, 3))
    x = tf.convert_to_tensor(x)
    canvas = batches.tf_batch_to_canvas(x)
    assert canvas.shape == (1, 300, 300, 3)

    canvas = batches.tf_batch_to_canvas(x, cols=5)
    assert canvas.shape == (1, 200, 500, 3)
    """
    if len(X.shape.as_list()) > 4:
        raise ValueError("input tensor has more than 4 dimensions.")
    N, H, W, C = X.shape.as_list()
    rc = math.sqrt(N)
    if cols is None:
        rows = cols = math.ceil(rc)
    else:
        cols = max(1, cols)
        rows = math.ceil(N / cols)
    n_white_tiles = cols * rows - N
    if n_white_tiles > 0:
        white_tiles = tf.ones((n_white_tiles, H, W, C), X.dtype)
        X = tf.concat([X, white_tiles], 0)
    image_shape = (H, W)
    n_channels = C
    return image_grid(X, (rows, cols), image_shape, n_channels)


# TODO(joelshor): Make this a special case of `image_reshaper`.
# shamelessly copied from https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/contrib/gan/python/eval/python/eval_utils_impl.py#L34-L80
def image_grid(input_tensor, grid_shape, image_shape=(32, 32), num_channels=3):
    """Arrange a minibatch of images into a grid to form a single image.

    Parameters
    ----------
    input_tensor :
        Tensor. Minibatch of images to format, either 4D
        ([batch size, height, width, num_channels]) or flattened
        ([batch size, height * width * num_channels]).
    grid_shape :
        Sequence of int. The shape of the image grid,
        formatted as [grid_height, grid_width].
    image_shape :
        Sequence of int. The shape of a single image,
        formatted as [image_height, image_width]. (Default value = (32)
    32) :

    num_channels :
         (Default value = 3)

    Returns
    -------

        Tensor representing a single image in which the input images have been

    Raises
    ------
    ValueError
        The grid shape and minibatch size don't match, or the image
        shape and number of channels are incompatible with the input tensor.

    """
    if grid_shape[0] * grid_shape[1] != int(input_tensor.shape[0]):
        raise ValueError(
            "Grid shape %s incompatible with minibatch size %i."
            % (grid_shape, int(input_tensor.shape[0]))
        )
    if len(input_tensor.shape) == 2:
        num_features = image_shape[0] * image_shape[1] * num_channels
        if int(input_tensor.shape[1]) != num_features:
            raise ValueError(
                "Image shape and number of channels incompatible with " "input tensor."
            )
    elif len(input_tensor.shape) == 4:
        if (
            int(input_tensor.shape[1]) != image_shape[0]
            or int(input_tensor.shape[2]) != image_shape[1]
            or int(input_tensor.shape[3]) != num_channels
        ):
            raise ValueError(
                "Image shape and number of channels incompatible with " "input tensor."
            )
    else:
        raise ValueError("Unrecognized input tensor format.")
    height, width = grid_shape[0] * image_shape[0], grid_shape[1] * image_shape[1]
    input_tensor = array_ops.reshape(
        input_tensor, tuple(grid_shape) + tuple(image_shape) + (num_channels,)
    )
    input_tensor = array_ops.transpose(input_tensor, [0, 1, 3, 2, 4])
    input_tensor = array_ops.reshape(
        input_tensor, [grid_shape[0], width, image_shape[0], num_channels]
    )
    input_tensor = array_ops.transpose(input_tensor, [0, 2, 1, 3])
    input_tensor = array_ops.reshape(input_tensor, [1, height, width, num_channels])
    return input_tensor
