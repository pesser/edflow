from edflow.data.dataset_mixin import DatasetMixin
from edflow.util import walk


class LateLoadingDataset(DatasetMixin):
    """The :class:`LateLoadingDataset` allows to work with examples containing
    `Callables`, which are evaluated by this Dataset. This way you can define
    data loading routines for images or other time consuming things in a base
    dataset, then add lots of data rearranging logic on top of this base
    dataset and in the end only load the subset of examples, you really want to
    use by calling the routines.

    .. code-block:: python

        class BaseDset:
            def get_example(self, idx):
                def _loading_routine():
                    load_image(idx)

                return {'image': _loading_routine}

        class AnchorDset:
            def __init__(self):
                B = BaseDset()

                self.S = SequenceDataset(B, 5)

            def get_example(self, idx):
                ex = self.S[idx]

                out = {}
                out['anchor1'] = ex['image'][0]
                out['anchor2'] = ex['image'][-1]

                return out


        final_dset = LateLoadingDataset(AnchorDset())


    """

    def __init__(self, base_dset):
        self.base_dset = base_dset

    def get_example(self, idx):
        base_ex = self.base_dset[idx]

        return walk(base_ex, expand, inplace=True, pass_key=False)


def expand(value):
    if callable(value):
        return value()
    return value
