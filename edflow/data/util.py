import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa
import matplotlib.gridspec as gridspec  # noqa

import numpy as np  # noqa
import cv2  # noqa

from edflow.util import walk  # noqa


def flow2hsv(flow):
    """Given a Flowmap of shape ``[W, H, 2]`` calculates an hsv image,
    showing the relative magnitude and direction of the optical flow.

    Args:
        flow (np.array): Optical flow with shape ``[W, H, 2]``.
    Returns:
        np.array: Containing the hsv data.
    """

    # prepare array - value is always at max
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 2] = 255

    # magnitude and angle
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # make it colorful
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    return hsv


def hsv2rgb(hsv):
    """color space conversion hsv -> rgb. simple wrapper for nice name."""
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def flow2rgb(flow):
    """converts a flow field to an rgb color image.

    Args:
        flow (np.array): optical flow with shape ``[W, H, 2]``.
    Returns:
        np.array: Containing the rgb data. Color indicates orientation,
            intensity indicates magnitude.
    """

    return hsv2rgb(flow2hsv(flow))


def get_support(image):
    """
    ... warning: This function makes a lot of assumptions that need not be met!

    Assuming that there are three categories of images and that the image_array
    has been properly constructed, this function will estimate the support of
    the given :attr:`image`.

    Args:
        image (np.ndarray): Some properly constructed image like array. No
            assumptions need to be made about the shape of the image, we
            simply assme each value is some color value.

    Returns:
        str: The support. Either '0->1', '-1->1' or '0->255'
    """

    if image.min() < 0:
        return "-1->1"
    elif image.max() > 1:
        return "0->255"
    else:
        return "0->1"


VALID_SUPPORTS = ["0->1", "-1->1", "0->255"]


def sup_str_to_num(support_str):
    """Converts a support string into usable numbers."""

    mn = -1.0 if support_str == "-1->1" else 0.0
    mx = 255.0 if support_str == "0->255" else 1.0

    return mn, mx


def adjust_support(image, future_support, current_support=None, clip=False):
    """Will adjust the support of all color values in :attr:`image`.

    Args:
        image (np.ndarray): Array containing color values. Make sure this is
            properly constructed.
        future_support (str): The support this array is supposed to have after
            the transformation. Must be one of '-1->1', '0->1', or '0->255'.
        current_support (str): The support of the colors currentl in
            :attr:`image`. If not given it will be estimated by
            :function:`get_support`.
        clip (bool): By default the return values in image are simply coming
            from a linear transform, thus the actual support might be larger
            than the requested interval. If set to ``True`` the returned
            array will be cliped to ``future_support``.

    Returns:
        same type as image: The given :attr:`image` with transformed support.
    """

    if current_support is None:
        current_support = get_support(image)
    else:
        assert current_support in VALID_SUPPORTS

    cur_min, cur_max = sup_str_to_num(current_support)
    fut_min, fut_max = sup_str_to_num(future_support)

    # To [0, 1]
    image = image.astype(float)
    image -= cur_min
    image /= cur_max - cur_min

    # To [fut_min, fut_max]
    image *= fut_max - fut_min
    image += fut_min

    if clip:
        image = clip_to_support(image, future_support)

    if future_support == "0->255":
        image = image.astype(np.uint8)

    return image


def clip_to_support(image, supp_str):
    vmin, vmax = sup_str_to_num(supp_str)

    return np.clip(image, vmin, vmax)


def add_im_info(image, ax):
    """Adds some interesting facts about the image."""

    shape = "x".join([str(s) for s in image.shape])
    mn = image.min()
    mx = image.max()

    supp = get_support(image)

    info_str = "shape: {}\nsupport: {} (min={}, max={})"
    info_str = info_str.format(shape, supp, mn, mx)

    ax.text(0, 0, info_str)


def im_fn(key, im, ax):
    """Plot an image. Used by :function:`plot_datum`."""
    if im.shape[-1] == 1:
        im = np.squeeze(im)

    add_im_info(im, ax)

    ax.imshow(adjust_support(im, "0->1"))
    ax.set_ylabel(key, rotation=0)


def heatmap_fn(key, im, ax):
    """Assumes that heatmap shape is [H, W, N]. Used by
    :function:`plot_datum`."""
    im = np.mean(im, axis=-1)
    im_fn(key, im, ax)


def flow_fn(key, im, ax):
    """Plot an flow. Used by :function:`plot_datum`."""
    im = flow2rgb(im)
    im_fn(key, im, ax)


def other_fn(key, obj, ax):
    """Print some text about the object. Used by :function:`plot_datum`."""
    text = "{}: {} - {}".format(key, type(obj), obj)
    ax.axis("off")
    # ax.imshow(np.ones([10, 100]))
    ax.text(0, 0, text)


PLOT_FUNCTIONS = {
    "image": im_fn,
    "heat": heatmap_fn,
    "flow": flow_fn,
    "other": other_fn,
}


def default_heuristic(key, obj):
    """Determines the kind of an object. Used by :function:`plot_datum`."""
    if isinstance(obj, np.ndarray):
        if len(obj.shape) > 3 or len(obj.shape) < 2:
            # This is no image -> Maybe later implement sequence fn
            return "other"
        else:
            if obj.shape[-1] in [3, 4]:
                return "image"
            elif obj.shape[-1] == 2:
                return "flow"
            else:
                return "heat"
    return "other"


def plot_datum(
    nested_thing,
    savename="datum.png",
    heuristics=default_heuristic,
    plt_functions=PLOT_FUNCTIONS,
):
    """Plots all data in the nested_thing as best as can.

    If heuristics is given, this determines how each leaf datum is converted
    to something plottable.

    Args:
        nested_thing (dict or list): Some nested object.
        savename (str): ``Path/to/the/plot.png``.
        heuristics (Callable): If given this should produce a string specifying
            the kind of data of the leaf. If ``None`` determinde automatically.
            See :function:`default_heuristic`.
        plt_functions (dict of Callables): Maps a ``kind`` to a function which
            can plot it. Each callable must be able to receive a the key, the
            leaf object and the Axes to plot it in.
    """

    class Plotter(object):
        def __init__(self, kind_fn, savename):
            self.kind_fn = kind_fn
            self.savename = savename
            self.buffer = []

        def __call__(self, key, obj):
            kind = self.kind_fn(key, obj)
            self.buffer += [[kind, key, obj]]

        def plot(self):
            n_pl = len(self.buffer)

            f = plt.figure(figsize=(5, 2 * n_pl))

            gs = gridspec.GridSpec(n_pl, 1)

            for i, [kind, key, obj] in enumerate(self.buffer):
                ax = f.add_subplot(gs[i])
                plt_functions[kind](key, obj, ax)

            f.savefig(self.savename)

        def __str__(self):
            self.plot()
            return "Saved Plot at {}".format(self.savename)

    P = Plotter(heuristics, savename)

    walk(nested_thing, P, pass_key=True)

    print(P)
