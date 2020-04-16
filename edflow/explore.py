import os
import sys
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
from edflow.data.dataset_mixin import DatasetMixin


def display_default(obj):
    """Find out how item could be displayed

    Parameters
    ----------
    obj : Any
        Item of example

    Returns
    -------
    str
        One of "Image", "Text", "Flow", "Segmentation", "None"
    """
    if isimage(obj):
        return "Image"
    elif istext(obj):
        return "Text"
    elif isflow(obj):
        return "Flow"
    elif issegmentation(obj):
        return "Segmentation"
    else:
        return "None"


def display(key, obj):
    """Display item in streamlit

    Parameters
    ----------
    key : str
        Subheader to be displayed
    obj : Any
        Item of example to be displayed
    """
    st.subheader(key)
    sel = selector(key, obj)
    if sel == "Text":
        st.text(obj)

    elif sel == "Image":
        st.image((obj + 1.0) / 2.0)

    elif sel == "Flow":
        display_flow(obj, key)

    elif sel == "Segmentation":
        idx = st.number_input("Segmentation Index", 0, obj.shape[2]-1, 0)
        img = obj[:,:,idx].astype(np.float)
        st.image(img)



def selector(key, obj):
    """Show select box to choose display mode of obj in streamlit

    Parameters
    ----------
    key : str
        Key of item to be displayed
    obj : Any
        Item to be displayed

    Returns
    -------
    str
        Selected display method for item
    """
    options = ["Auto", "Text", "Image", "Flow", "Segmentation", "None"]
    idx = options.index(display_default(obj))
    select = st.selectbox("Display {} as".format(key), options, index=idx)
    return select


def custom_visualizations(ex, config):
    """Displays custom visualizations in streamlit

    The visualizations can be inserted to the config via their import path.
    Everyone can implement a custom visualization for an example.

    The visualization functions must accept the example and config as positional
    arguments.


    Examples
    --------

    Add visualizations to the text box with their import path. For example:

    .. code-block:: python

        edflow.util.edexplore.display_flow_on_image


    A valid visualization function could look like for example:

    .. code-block:: python

        import streamlit as st
        from edflow.util.edexplore import isimage, st_get_list_or_dict_item

        def my_visualization(ex, config):
            st.write("This is my visualization")

            image1, image1_key = st_get_list_or_dict_item(ex, "image1", filter_fn=isimage)

            st.image((image1 + 1) / 2)
            
            image2 = ex["image2"]
            image3 = ex["image3"]

            st.image((image2 + 1) / 2)
            st.image((image3 + 1) / 2)


    Visualizations can be displayed by default, if they are specified in the config.
    An example for the configuration yaml file would be:

    .. code-block::

        edexplore:
            visualizations:
                custom:
                    vis1:
                        path: my_package.visualizations.my_visualization


    Parameters
    ----------
    ex : dict
        Example to be visualized
    config : dict
        Edexplore config
    """
    st.header("Custom visualizations")
    default_visualizations = retrieve(
        config, "edexplore/visualizations/custom", default=dict()
    )
    import_paths = [
        vis.get("path", None)
        for vis in default_visualizations.values()
        if isinstance(vis.get("path", None), str)
    ]
    visualizatins_str = st.text_input(
        "Visualization import paths: comma separated", value=",".join(import_paths)
    )

    if len(visualizatins_str) == 0:
        st.text(custom_visualizations.__doc__)

    for vis in [vis for vis in visualizatins_str.split(",") if vis != ""]:
        try:
            st.subheader(vis)
            impl = get_obj_from_str(vis, reload=True)
            impl(ex, config)
        except Exception as error:
            st.write(error)


ADDITIONAL_VISUALIZATIONS = {
    "custom": custom_visualizations,
    "optical_flow_on_image": display_flow_on_image,
}


def show_example(dset, idx, config):
    """Show example of dataset

    Parameters
    ----------
    dset : DatasetMixin
        Dataset to be shown
    idx : int
        Index to be shown
    config : dict
        Config used to show example
    """

    ex = dset[idx]

    # additional visualizations
    default_additional_visualizations = retrieve(
        config, "edexplore/visualizations", default=dict()
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
    """Explore dataset specified in config

    Parameters
    ----------
    config : dict
        Edflow config dict used to explore dataset
    disable_cache : bool, optional
        Disable cache while loading dataset, by default False
    """
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
        idx = st.sidebar.slider("Index", 0, len(dset)-1, 0)
    elif input_method == "Number input":
        idx = st.sidebar.number_input("Index", 0, len(dset)-1, 0)
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
