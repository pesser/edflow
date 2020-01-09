import os
import numpy as np

from edflow.data.believers.meta import MetaDataset
from edflow.data.believers.meta_util import store_label_mmap
from edflow.util import retrieve, get_obj_from_str, walk

from tqdm.autonotebook import tqdm


class MetaViewDataset(MetaDataset):
    """The :class:`MetaViewDataset` implements a way to render out a view of a
    base dataset without the need to rewrite/copy the load heavy data in the
    base dataset.

    To use the MetaViewDataset you need to define two things:
        1. A base dataset as import string in the ``meta.yaml`` file. Use the
            key ``base_dset`` for this. This should preferably be a function or
            class, which is passed the kwargs ``base_kwargs`` as defined in the
            ``meta.yaml``..
        2. A view in the form of a numpy ``memmap`` or a nested object of
            ``dict``s and ``list``s with ``memmaps`` at the leaves, each
            storing the indices used for the view in this dataset. The arrays
            can be of any dimensionality, but no value must be outside the
            range ``[0, len(base dataset)]`` and they must all be of the same
            length.

    The dimensionality of the view is reflected in the nestednes of the
    resulting examples.

    ### Example

    You have a base dataset, which contains video frames. It has length ``N``.

    Say you want to have a combination of two views on your dataset: One
    contains all ``M`` possible subsequences of length 5 of videos contained in
    the dataset and one contains an appearance image per each example with the
    same person as in the sequence.
    
    All you need is to define two numpy arrays, one with the indices belonging
    to the sequenced frames and one with indices of examples of the appearence
    images. They should look something like this:

    .. code-block:: python

        # Sequence indices
        seq_idxs = [[0, 1, 2, 3, 4],
                    [1, 2, 3, 4, 5],
                    [2, 3, 4, 5, 6],
                    [3, 4, 5, 6, 7],
                    ...
                    [N-4, N-3, N-2, N-1, N],
        print(seq_idxs.shape)  # [M, 5]

        # Sequence indices
        app_idxs = [12,
                    12,
                    15,
                    10,
                    ..
                    109],
        print(app_idxs.shape)  # [M]

    Knowing your views, create a folder, where you want to store your view
    dataset, i.e. at some path ``ROOT``. Create a folder ``ROOT/labels`` and
    store the views according to the label naming scheme as defined in the
    :class:`MetaDataset`. You can use the function
    :func:`edflow.data.believers.meta_util.store_label_mmap` for this.
    You can also store the views in any subfolder of labels, which might come
    in handy if you have a lot of labels and want to keep things clean.

    Finally create a file ``ROOT/meta.yaml``.

    Our folder should look something like this:

    .. code-block:: bash

        ROOT/
         ├ labels/
         │ ├ app_view-*-{M}-*-int64.npy
         │ └ seq_view-*-{M}x5-*-int64.npy
         └ meta.yaml

    Now let us fill the ``meta.yaml``. All we need to do is specify the base
    dataset and how we want to use our views:

    .. code-block:: yaml

        # meta.yaml

        description: |
            This is our very own View on the data.
            Let's have fun with it!

        base_dset: import.path.to.dset_object
        base_kwargs:
            stuff: needed_for_construction

        views:
            appearance: app_view
            frames: seq_view

    Now we are ready to construct our view on the base dataset!
    Use ``.show()`` to see how the dataset looks like. This works especially
    nice in a jupyter notebook.

    .. code-block:: python

        ViewDset = MetaViewDataset('ROOT')

        print(ViewDset.labels.keys())  # ['appearance', 'frames']
        print(len(ViewDset))  # {M}

        ViewDset.show()  # prints the labels and the first example
    """

    def __init__(self, root):
        super().__init__(root)

        base_import = retrieve(self.meta, "base_dset")
        base_kwargs = retrieve(self.meta, "base_kwargs")
        self.base = get_obj_from_str(base_import)(**base_kwargs)
        self.base.append_labels = False

        views = retrieve(self.meta, "views", default="view")

        def get_label(key):
            return retrieve(self.labels, key)

        self.views = walk(views, get_label)

        if not os.path.exists(os.path.join(root, ".constructed.txt")):

            def constructor(name, view):
                folder_name = name
                savefolder = os.path.join(root, "labels", folder_name)

                os.makedirs(savefolder, exist_ok=True)

                for key, label in tqdm(
                    self.base.labels.items(), desc=f"Exporting View {name}"
                ):

                    savepath = os.path.join(root, "labels", name)
                    label_view = np.take(label, view, axis=0)
                    store_label_mmap(label_view, savepath, key)

            walk(self.views, constructor, pass_key=True)

            with open(os.path.join(root, ".constructed.txt"), "w+") as cf:
                cf.write(
                    "Do not delete, this reduces loading times.\n"
                    "If you need to re-render the view, you can safely "
                    "delete this file."
                )

            # Re-initialize as we need to load the labels again.
            super().__init__(root)

    def get_example(self, idx):
        """Get the examples from the base dataset at defined at ``view[idx]``. Load loaders if applicable.
        """

        def get_view(view):
            return view[idx]

        view = walk(self.views, get_view)

        view_example = walk(view, self.base.__getitem__, walk_np_arrays=True)

        if len(self.loaders) > 0:
            loaders_example = super().get_example(idx)
            view_example.update(loaders_example)

        return view_example
