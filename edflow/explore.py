import os
import sys
import random
import argparse
import yaml
import streamlit as st
from edflow.config import parse_unknown_args, update_config

import numpy as np
from edflow.util import walk, pp2mkdtable, retrieve, get_leaf_names
from edflow import get_obj_from_str


def isimage(obj):
    return (
        isinstance(obj, np.ndarray)
        and len(obj.shape) == 3
        and obj.shape[2] in [1, 3, 4]
    )


def isflow(obj):
    return isinstance(obj, np.ndarray) and len(obj.shape) == 3 and obj.shape[2] in [2]


def istext(obj):
    return isinstance(obj, (int, float, str, np.integer, np.float))


def display_default(obj):
    if isimage(obj):
        return "Image"
    elif istext(obj):
        return "Text"
    elif isflow(obj):
        return "Flow"
    else:
        return "None"


def display(key, obj):
    st.subheader(key)
    sel = selector(key, obj)
    if sel == "Text":
        st.text(obj)

    elif sel == "Image":
        st.image((obj + 1.0) / 2.0)

    elif sel == "Flow":
        import flowiz as fz

        img = fz.convert_from_flow(obj)
        st.image(img)

        import matplotlib.pyplot as plt

        magnitudes = np.sqrt(obj[:, :, 0] ** 2 + obj[:, :, 1] ** 2).reshape(-1)

        fig, ax = plt.subplots(1, 1)
        ax.hist(magnitudes, log=True, label="magnitudes", alpha=0.7)
        ax.hist(obj[:, :, 0].reshape(-1), log=True, label="flow[:,:,0]: dx", alpha=0.7)
        ax.hist(obj[:, :, 1].reshape(-1), log=True, label="flow[:,:,1]: dy", alpha=0.7)
        ax.set_title("flow values")
        ax.legend()

        st.pyplot(fig)


def first_index(keys, key_part):
    for i, key in enumerate(keys):
        if key_part in key:
            return i
    return 0


def display_flow_on_image(ex, config):
    import matplotlib.pyplot as plt
    from skimage.transform import downscale_local_mean

    st.subheader("Optical flow on image")

    # get user input
    example_keys = get_leaf_names(ex)
    image_keys = [key for key in example_keys if isimage(retrieve(ex, key, default=0))]
    flow_keys = [key for key in example_keys if isflow(retrieve(ex, key, default=0))]
    subconfig = retrieve(
        config,
        "edexplore/additional_visualizations/optical_flow_on_image",
        default=dict(),
    )
    image_search_key = retrieve(subconfig, "image_key", default="image")
    flow_search_key = retrieve(subconfig, "flow_key", default="flow")
    default_vector_frequency = retrieve(subconfig, "vector_frequency", default=4)
    image_key = st.selectbox(
        "Image key", image_keys, index=first_index(image_keys, image_search_key)
    )
    flow_key = st.selectbox(
        "Flow key", flow_keys, index=first_index(flow_keys, flow_search_key)
    )
    freq = st.number_input(
        "Flow vector every ... pixels", value=default_vector_frequency, min_value=1
    )

    # get image, X, Y, U and V
    image = retrieve(ex, image_key)
    flow = retrieve(ex, flow_key)
    H, W = flow.shape[:2]
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    U, V = flow[:, :, 0], flow[:, :, 1]

    # use only samples, where mean is computed completely from within the original image range
    sample_height = H // freq
    sample_width = W // freq

    # average flow values locally
    X = downscale_local_mean(X, (freq, freq))[:sample_height, :sample_width]
    Y = downscale_local_mean(Y, (freq, freq))[:sample_height, :sample_width]
    U = downscale_local_mean(U, (freq, freq))[:sample_height, :sample_width]
    V = downscale_local_mean(V, (freq, freq))[:sample_height, :sample_width]

    # plot image and flow on figure
    fig, ax = plt.subplots(1, 1)
    ax.set_title(image_key + " and " + flow_key)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.imshow((image + 1.0) / 2.0)
    ax.quiver(X, Y, U, V, color="r", angles="xy", scale_units="xy", scale=1)

    # show data on streamlit
    st.pyplot(fig)
    display(flow_key, flow)


def selector(key, obj):
    options = ["Auto", "Text", "Image", "Flow", "None"]
    idx = options.index(display_default(obj))
    select = st.selectbox("Display {} as".format(key), options, index=idx)
    return select


ADDITIONAL_VISUALIZATIONS = {
    "optical_flow_on_image": display_flow_on_image,
}


def show_example(dset, idx, config):

    ex = dset[idx]

    # additional visualizations
    default_additional_visualizations = retrieve(
        config, "edexplore/additional_visualizations", default=dict()
    ).keys()
    additional_visualizations = st.sidebar.multiselect(
        "Additional visualizations",
        list(ADDITIONAL_VISUALIZATIONS.keys()),
        default=default_additional_visualizations,
    )
    if len(additional_visualizations) > 0:
        st.header("Additional visualizations")
        for key in additional_visualizations:
            ADDITIONAL_VISUALIZATIONS[key](ex, config)

    # dataset items
    st.header("Keys")
    walk(ex, display, pass_key=True)

    # summaries
    st.header("Summary")
    summary = pp2mkdtable(ex, jupyter_style=True)
    # print markdown summary on console for easy copy and pasting in readme etc
    print(summary)
    st.markdown(summary)


def _get_state(config):
    Dataset = get_obj_from_str(config["dataset"])
    dataset = Dataset(config)
    return dataset


def explore(config, disable_cache=False):
    if not disable_cache:
        get_state = st.cache(persist=False, allow_output_mutation=True)(_get_state)
    else:
        get_state = _get_state
    dset = get_state(config)
    dset.expand = True
    st.title("Dataset Explorer: {}".format(type(dset).__name__))

    input_method = st.sidebar.selectbox(
        "Index selection method", ["Slider", "Number input", "Sample"]
    )
    if input_method == "Slider":
        idx = st.sidebar.slider("Index", 0, len(dset), 0)
    elif input_method == "Number input":
        idx = st.sidebar.number_input("Index", 0, len(dset), 0)
    elif input_method == "Sample":
        idx = 0
        if st.sidebar.button("Sample"):
            idx = np.random.choice(len(dset))
        st.sidebar.text("Index: {}".format(idx))

    show_example(dset, idx, config)

    st.header("config")
    cfg_string = pp2mkdtable(config, jupyter_style=True)
    cfg = st.markdown(cfg_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="edflow dataset explorer")
    parser.add_argument(
        "-d",
        "--disable_cache",
        action="store_true",
        help="Disable caching dataset instantiation.",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="config.yaml",
        help="Paths to base configs.",
        default=list(),
    )
    try:
        sys.path.append(os.getcwd())
        opt, unknown = parser.parse_known_args()
        additional_kwargs = parse_unknown_args(unknown)
        config = dict()
        for base_config in opt.base:
            with open(base_config) as f:
                config.update(yaml.full_load(f))
        update_config(config, additional_kwargs)
    except SystemExit as e:
        # This exception will be raised if --help or invalid command line arguments
        # are used. Currently streamlit prevents the program from exiting normally
        # so we have to do a hard exit.
        os._exit(e.code)

    explore(config, disable_cache=opt.disable_cache)
