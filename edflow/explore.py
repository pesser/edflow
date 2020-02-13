import os
import sys
from typing import Any
import random
import argparse
import yaml
import streamlit as st
from edflow.config import parse_unknown_args, update_config

import numpy as np
from edflow.util import walk, pp2mkdtable, retrieve, get_leaf_names
from edflow.util.edexplore import (
    isimage,
    isflow,
    istext,
    display_flow,
    display_flow_on_image,
)
from edflow import get_obj_from_str


def display_default(obj: Any) -> str:
    """Find out how item could be displayed

    Parameters
    ----------
    obj : Any
        Item of example

    Returns
    -------
    str
        One of "Image", "Text", "Flow", "None"
    """
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
        display_flow(obj, key)


def selector(key, obj):
    options = ["Auto", "Text", "Image", "Flow", "None"]
    idx = options.index(display_default(obj))
    select = st.selectbox("Display {} as".format(key), options, index=idx)
    return select


def custom_visualizations(ex, config):
    st.header("Custom visualizations")
    default_visualizations = retrieve(
        config, "edexplore/custom_visualizations", default=dict()
    )
    st.text(default_visualizations)
    import_paths = [
        vis.get("path", None)
        for vis in default_visualizations.values()
        if isinstance(vis.get("path", None), str)
    ]
    visualizatins_str = st.text_input(
        "Visualization import paths: comma separated", value=",".join(import_paths)
    )

    for vis in visualizatins_str.split(","):
        try:
            st.subheader(vis)
            impl = get_obj_from_str(vis)
            impl(ex, config)
        except Exception as error:
            st.text(error)


ADDITIONAL_VISUALIZATIONS = {
    "optical_flow_on_image": display_flow_on_image,
    "custom_visualizations": custom_visualizations,
}


def show_example(dset, idx, config):

    ex = dset[idx]

    # additional visualizations
    default_additional_visualizations = retrieve(
        config, "edexplore", default=dict()
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
