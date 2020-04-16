import streamlit as st
import numpy as np

from edflow.util import walk, pp2mkdtable, retrieve, get_leaf_names


def isimage(obj):
    """Heuristic if item could be displayed as an image

    Parameters
    ----------
    obj : Any
        item

    Returns
    -------
    bool
        Returns True for rank-three numpy arrays where third axis is one,
        three or, four. And for rank-two arrays if both sides are bigger than 4.
    """
    return isinstance(obj, np.ndarray) and (
        (len(obj.shape) == 3 and obj.shape[2] in [1, 3, 4])
        or (len(obj.shape) == 2 and all([size > 4 for size in obj.shape[:2]]))
    )


def isflow(obj):
    """Heuristic if item could be displayed as an optical flow field

    Parameters
    ----------
    obj : Any
        item

    Returns
    -------
    bool
        Returns True for rank-three numpy arrays where third axis is two.
    """
    return isinstance(obj, np.ndarray) and len(obj.shape) == 3 and obj.shape[2] in [2]


def issegmentation(obj):
    """Heuristic if item could be displayed as a segmentation mask

    Parameters
    ----------
    obj : Any
        item

    Returns
    -------
    bool
        Returns True for rank-three numpy arrays with values boolean or in
        [0,1]
    """
    is_ = isinstance(obj, np.ndarray) and len(obj.shape) == 3
    is_ = is_ and ((obj.dtype == np.bool) or
                   (0.0 <= obj.min() and obj.max() <= 1.0))
    return is_


def istext(obj):
    """Heuristic if item could be displayed as text

    Parameters
    ----------
    obj : Any
        item

    Returns
    -------
    bool
        Retruns True for items of type int, float, str, np.integer, np.float
    """
    return isinstance(obj, (int, float, str, np.integer, np.float))


def display_flow(obj, key="flow values"):
    """Displays flow colored image and histogram in streamlit

    Parameters
    ----------
    obj : np.ndarray
        Optical flow field
    key : str, optional
        Title of the histogram, by default "flow values"
    """
    import flowiz as fz

    img = fz.convert_from_flow(obj)
    st.image(img)

    import matplotlib.pyplot as plt

    magnitudes = np.sqrt(obj[:, :, 0] ** 2 + obj[:, :, 1] ** 2).reshape(-1)

    fig, ax = plt.subplots(1, 1)
    ax.hist(magnitudes, log=True, label="magnitudes", alpha=0.7)
    ax.hist(obj[:, :, 0].reshape(-1), log=True, label="flow[:,:,0]: dx", alpha=0.7)
    ax.hist(obj[:, :, 1].reshape(-1), log=True, label="flow[:,:,1]: dy", alpha=0.7)
    ax.set_title(key)
    ax.legend()

    st.pyplot(fig)


def _max_flow_magnitude(blocks, axis):
    """Method, which downsamples a flow vector field. To be used with skimage.measure.block_reduce.
    """
    assert axis == (3, 4, 5)
    assert blocks.shape[2] == 1 and blocks.shape[5] == 2

    num_blocks = np.multiply.reduce(blocks.shape[:3])
    pixels_per_block = np.multiply.reduce(blocks.shape[3:-1])
    blocks_flat = np.reshape(blocks, (num_blocks, pixels_per_block, 2))

    max_vectors = [max(block_flat, key=np.linalg.norm) for block_flat in blocks_flat]
    # max_vectors = list(map(lambda block_flat: max(block_flat, key=np.linalg.norm), blocks_flat))
    downsampled = np.reshape(np.stack(max_vectors), (*blocks.shape[:2], 2))

    return downsampled


flow_downsample_methods = {
    "mean": np.mean,
    "median": np.median,
    "max_magnitude": _max_flow_magnitude,
}


def first_index(keys, key_part):
    """Find first item in iterable containing part of the string

    Parameters
    ----------
    keys : Iterable[str]
        Iterable with strings to search through
    key_part : str
        String to look for

    Returns
    -------
    int
        Returns index of first element in keys containing key_part, 0 if not found.
    """
    for i, key in enumerate(keys):
        if key_part in key:
            return i
    return 0


def st_get_list_or_dict_item(
    list_or_dict,
    item_key,
    description=None,
    filter_fn=lambda ex_item: True,
    config=None,
    config_key=None,
    selectbox_key=None,
):
    """Displays streamlit selectbox for selecting a value by key from list or dict

    Parameters
    ----------
    list_or_dict : Union[list, dict]
        List or dict to find item in
    item_key : str
        Key how item too look for is most likely called
    description : str, optional
        Description for streamlit selectbox, by default last part of config_key or item key
    filter_fn : callable, optional
        Function to check if item is desired, by default `lambda ex_item:True`
    config : dict, optional
        Config for default values, by default None
    config_key : str, optional
        How to find default value in config, by default None
    selectbox_key : str, optional
        Key passed to streamlit.selectbox

    Returns
    -------
    Tuple[Any, str]
        item, item_key for item found in dict or list
    """
    if description is None:
        if config_key is not None:
            description = config_key.split("/")[-1]
        else:
            description = item_key
    # Look for key in config
    if config is not None and config_key is not None:
        item_key = retrieve(config, config_key, default=item_key)
    # find all list_or_dict items matching filter_function
    list_or_dict_keys = get_leaf_names(list_or_dict)
    item_keys = [
        key
        for key in list_or_dict_keys
        if filter_fn(retrieve(list_or_dict, key, default=0, expand=False))
    ]
    # display selectbox
    item_key = st.selectbox(
        description,
        item_keys,
        index=first_index(item_keys, item_key),
        key=selectbox_key,
    )
    # get item
    item = retrieve(list_or_dict, item_key, expand=False)
    return item, item_key


def display_flow_on_image(ex, config):
    """Display flow vectors on image in streamlit

    Add config for this visualization to your config file to enable this
    visualization by default.


    Examples
    --------

    Add visualizations to the text box with their import path. For example:

    .. code-block::

        edexplore:
            visualizations:
                optical_flow_on_image:
                    image_key: "images/0/image"
                    flow_key: "forward_flow"
                    vector_frequency: 5
                    flow_downsample_method: max_magnitude

    Parameters
    ----------
    ex : dict
        Example dict from dataset
    config : dict
        Config dict
    """

    st.subheader("Optical flow on image")

    # get user input
    image, image_key = st_get_list_or_dict_item(
        ex,
        "image",
        filter_fn=isimage,
        config=config,
        config_key="edexplore/visualizations/optical_flow_on_image/image_key",
    )
    flow, flow_key = st_get_list_or_dict_item(
        ex,
        "flow",
        filter_fn=isflow,
        config=config,
        config_key="edexplore/visualizations/optical_flow_on_image/flow_key",
    )
    downsample_method, downsample_method_key = st_get_list_or_dict_item(
        flow_downsample_methods,
        "flow_downsample_method",
        config=config,
        config_key="edexplore/visualizations/optical_flow_on_image/flow_downsample_method",
    )
    freq = st.number_input(
        "vector_frequency",
        value=retrieve(
            config,
            "edexplore/visualizations/optical_flow_on_image/vector_frequency",
            default=4,
        ),
        min_value=1,
    )

    import matplotlib.pyplot as plt
    from skimage.transform import downscale_local_mean
    from skimage.measure import block_reduce

    # get image, X, Y
    H, W = flow.shape[:2]
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    # use only samples, where mean is computed completely from within the original image range
    sample_height = H // freq
    sample_width = W // freq

    # average flow values locally
    X = downscale_local_mean(X, (freq, freq))[:sample_height, :sample_width]
    Y = downscale_local_mean(Y, (freq, freq))[:sample_height, :sample_width]

    if downsample_method_key == "max_magnitude":
        flow_downsampled = block_reduce(flow, (freq, freq, 2), downsample_method)[
            :sample_height, :sample_width
        ]
    else:
        flow_downsampled = block_reduce(flow, (freq, freq, 1), downsample_method)[
            :sample_height, :sample_width
        ]

    U, V = flow_downsampled[:, :, 0], flow_downsampled[:, :, 1]

    # plot image and flow on figure
    fig, ax = plt.subplots(1, 1)
    ax.set_title(
        "[{}] {}\nand {} {} arrows".format(
            retrieve(ex, "index_", default=""),
            image_key,
            flow_key,
            downsample_method_key,
        )
    )
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.imshow((image + 1.0) / 2.0)
    ax.quiver(X, Y, U, V, color="r", angles="xy", scale_units="xy", scale=1)
    # show data on streamlit
    st.pyplot(fig)
    display_flow(flow, "flow values")
    display_flow(
        flow_downsampled, "flow arrows {} downsampled".format(downsample_method_key)
    )
