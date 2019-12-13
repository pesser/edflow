"""
Datasets TLDR
=============

Datasets contain examples, which can be accessed by an index::

    example = Dataset[index]

Each example is annotated by labels. These can be accessed via the
:attr:`labels` attribute of the dataset::

    label = Dataset.labels[key][index]

To make a working dataset you need to implement a :meth:`get_example` method, which must return a ``dict``,
a :meth:`__len__` method and define the :attr:`labels` attribute, which must
be a dict, that can be empty.

.. warning::

    Dataset, which are specified in the edflow config must accept one
    positional argument ``config``!

If you have to worry about dataloading take a look at the
:class:`LateLoadingDataset`. You can define datasets to return examples
containing callables for heavy dataloading, which are only executed by the
:class:`LateLoadingDataset`. Having this class as the last in your dataset
pipline can potentially speed up your data loading.
"""

# TODO maybe just pull
# https://github.com/chainer/chainer/blob/v4.4.0/chainer/dataset/dataset_mixin.py
# into the rep to avoid dependency on chainer for this one mixin - it doesnt
# even do that much and it would provide better documentation as this is
# actually our base class for datasets

from edflow.util import PRNGMixin


# Legacy imports
# TODO: Remove these legacy imports at the next release
from edflow.data.dataset_mixin import DatasetMixin

from edflow.data.agnostics.subdataset import SubDataset
from edflow.data.agnostics.concatenated import ConcatenatedDataset
from edflow.data.agnostics.concatenated import ExampleConcatenatedDataset
from edflow.data.agnostics.concatenated import DisjunctExampleConcatenatedDataset
from edflow.data.agnostics.csv_dset import CsvDataset

from edflow.data.believers.sequence import SequenceDataset, UnSequenceDataset
from edflow.data.believers.sequence import getSeqDataset

from edflow.data.processing.labels import LabelDataset, ExtraLabelsDataset
from edflow.data.processing.processed import ProcessedDataset

from edflow.data.util.cached_dset import *
from edflow.data.util.util_dsets import *


if __name__ == "__main__":
    # import matplotlib.pyplot as plt

    # r = '/home/johannes/Documents/Uni HD/Dr_J/Projects/data_creation/' \
    #     'show_vids/cut_selection/fortnite/'

    # def rfn(im_path):
    #     return {'image': plt.imread(im_path)}

    # def lfn(path):
    #     if os.path.isfile(path) and path[-4:] == '.jpg':
    #         fname = os.path.basename(path)
    #         labels = fname[:-4].split('_')
    #         if len(labels) == 3:
    #             pid, act, fid = labels
    #             beam = False
    #         else:
    #             pid, act, _, fid = labels
    #             beam = True
    # return {'pid': int(pid), 'vid': 0, 'fid': int(fid), 'action': act}

    # D = DataFolder(r,
    #                rfn,
    #                lfn,
    #                ['pid', 'vid', 'fid'])

    # for i in range(10):
    #     d = D[i]
    #     print(',\n '.join(['{}: {}'.format(k, v if not hasattr(v, 'shape')
    #                                        else v.shape)
    #                                        for k, v in d.items()]))

    from edflow.debug import DebugDataset

    D = DebugDataset()

    def labels(data, i):
        return {"fid": i}

    D = ExtraLabelsDataset(D, labels)
    print("D")
    for k, v in D.labels.items():
        print(k)
        print(np.shape(v))

    S = SequenceDataset(D, 2)
    print("S")
    for k, v in S.labels.items():
        print(k)
        print(np.shape(v))

    S = SubDataset(S, [2, 5, 10])
    print("Sub")
    for k, v in S.labels.items():
        print(k)
        print(np.shape(v))

    U = UnSequenceDataset(S)
    print("U")
    for k, v in U.labels.items():
        print(k)
        print(np.shape(v))

    print(len(S))
    print(U.seq_len)
    print(len(U))

    for i in range(len(U)):
        print(U[i])
