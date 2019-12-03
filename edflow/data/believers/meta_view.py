import os
import numpy as np

from edflow.data.believers.meta import MetaDataset
from edflow.data.believers.meta_util import store_label_mmap
from edflow.util import retrieve, get_obj_from_str, walk

from tqdm.autonotebook import tqdm


class MetaViewDataset(MetaDataset):
    '''The :class:`MetaViewDataset` implements a way to render out a view of a
    base dataset without the need to rewrite/copy the load heavy data in the
    base dataset.

    To use the MetaViewDataset you need to define two things:
        1. A base dataset as import string in the ``meta.yaml`` file. Use the
            key ``base_dset`` for this. This should preferably be a function or
            object, which is passed the args and kwargs ``base_args`` and
            ``base_kwargs``.
        2. A view in the form of a numpy ``memmap`` or a nested object of
            ``dict``s and ``list``s with ``memmaps`` at the leaves, each
            storing the indices used for the view in this dataset. The arrays
            can be of any dimensionality, but no value must be outside the
            range ``[0, len(base dataset)]`` and they must all be of the same
            length.

    The dimensionality of the view is reflected in the nestednes of the
    resulting examples.
    '''

    def __init__(self, root, *base_args, **base_kwargs):
        super().__init__(root)

        base_import = retrieve(self.meta, 'base_dset')
        self.base = get_obj_from_str(base_import)(*base_args, **base_kwargs)
        self.base.append_labels = False

        views = retrieve(self.meta, 'views', default='view')

        def get_label(key):
            return retrieve(self.labels, key)
        self.views = walk(views, get_label)

        if not os.path.exists(os.path.join(root, '.constructed.txt')):
            def constructor(name, view):
                folder_name = name
                savefolder = os.path.join(root, 'labels', folder_name)

                os.makedirs(savefolder, exist_ok=True)

                for key, label in tqdm(self.base.labels.items(),
                                       desc=f'Exporting View {name}'):

                    savepath = os.path.join(root, 'labels', name)
                    label_view = np.take(label, view, axis=0)
                    store_label_mmap(label_view,
                                     savepath,
                                     key)
            walk(self.views, constructor, pass_key=True)

            with open(os.path.join(root, '.constructed.txt'), 'w+') as cf:
                cf.write('Do not delete, this reduces loading times.\n'
                         'If you need to re-render the view, you can safely '
                         'delete this file.')

            # Re-initialize as we need to load the labels again.
            super().__init__(root)

    def get_example(self, idx):
        """Get the examples from the base dataset at defined at ``view[idx]``.
        """

        def get_view(view):
            return view[idx]

        view = walk(self.views, get_view)

        view_example = walk(view, self.base.__getitem__, walk_np_arrays=True)

        return view_example
