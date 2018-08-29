import numpy as np
import PIL.Image
import numpy as np
import scipy.ndimage
import math
import pickle
import os
import random
from edflow.keypointutils import make_heatmaps, make_stickman
from edflow.keypointutils import make_crops, reorder_keypoints
from chainer.dataset import DatasetMixin
from chainer.iterators import MultiprocessIterator


def load_image(path):
    img = PIL.Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    x = np.asarray(img, dtype = "float32")
    x = x / 127.5 - 1.0
    return x


def save_image(x, path):
    """Save image."""
    x = (x + 1.0) / 2.0
    x = np.clip(255 * x, 0, 255)
    x = np.array(x, dtype = "uint8")
    PIL.Image.fromarray(x).save(path)


def resize(x, size):
    try:
        zoom = [size[0] / x.shape[0],
                size[1] / x.shape[1]]
    except TypeError:
        zoom = [size / x.shape[0],
                size / x.shape[1]]
    for _ in range(len(x.shape)-2):
        zoom.append(1.0)
    x = scipy.ndimage.zoom(x, zoom)
    return x


def tile(X, rows, cols):
    """Tile images for display."""
    tiling = np.zeros((rows * X.shape[1], cols * X.shape[2], X.shape[3]), dtype = X.dtype)
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < X.shape[0]:
                img = X[idx,...]
                tiling[
                        i*X.shape[1]:(i+1)*X.shape[1],
                        j*X.shape[2]:(j+1)*X.shape[2],
                        :] = img
    return tiling


def plot_batch(X, out_path):
    """Save batch of images tiled."""
    n_channels = X.shape[3]
    if n_channels > 4:
        X = X[:, :, :, :3]
    if n_channels == 1:
        X = np.tile(X, [1, 1, 1, 3])
    rc = math.sqrt(X.shape[0])
    rows = cols = math.ceil(rc)
    canvas = tile(X, rows, cols)
    save_image(canvas, out_path)


def subindex(index, indices):
    new_index = dict()
    new_index["joint_order"] = index["joint_order"]
    for k in index._keys:
        new_index[k] = [index[k][i] for i in indices]
    return new_index


class Index(object):
    """
    Interface to data.
    The index pickle file should be a dictionary with keys
        joint_order: List of joint names
        pid: List of person ids
        vid: List of video ids per person
        fid: List of frame ids per video
        image: List of relative paths to image
        keypoints: List of relative paths to keypoints (.npy) in [0,1] or
            negative for invalid ones
        mask: List of relative paths to mask (.npy)
    """
    joint_order = [
            'rankle', 'rknee', 'rhip', 'lhip', 'lknee', 'lankle',
            'rwrist', 'relbow', 'rshoulder', 'lshoulder', 'lelbow',
            'lwrist', 'cnose', 'leye', 'reye']

    def __init__(self, path):
        self.path = path
        self.root = os.path.dirname(path)
        with open(self.path, "rb") as f:
            self._index = pickle.load(f)
        self._keys = ["pid", "vid", "fid", "image", "keypoints", "mask"]
        self._len = len(self._index["image"])
        for k in self._keys:
            assert len(self._index[k]) == self._len
        self.loaders = {
                "pid": lambda x: self["pid"][x],
                "vid": lambda x: self["vid"][x],
                "fid": lambda x: self["fid"][x],
                "image": self.load_image,
                "keypoints": self.load_keypoints,
                "mask": self.load_mask}
        self.pids = set(self["pid"])


    def where(self, pid = None, vid = None, fid = None):
        indices = [i for i in range(len(self))]
        for k,v in [("pid",pid), ("vid",vid), ("fid",fid)]:
            if v is not None:
                indices = [i for i in indices if self._index[k][i] == v]
        return indices


    def __len__(self):
        return self._len


    def __getitem__(self, i):
        return self._index[i]


    def get_index(self, pid, vid, fid):
        check = lambda i: self["pid"][i] == pid and self["vid"][i] == vid and self["fid"][i] == fid
        idx = [i for i in range(len(self)) if check(i)]
        if not len(idx) == 1:
            raise KeyError("{}, {}, {} resulted in {}".format(pid,vid,fid,idx))
        return idx[0]


    def load_image(self, i):
        path = os.path.join(self.root, self._index["image"][i])
        return load_image(path)


    def load_keypoints(self, i):
        path = os.path.join(self.root, self._index["keypoints"][i])
        keypoints = np.float32(np.load(path))
        keypoints = reorder_keypoints(keypoints,
                src_order = self._index["joint_order"],
                dst_order = self.joint_order)
        return keypoints


    def load_mask(self, i):
        path = os.path.join(self.root, self._index["mask"][i])
        return np.float32(np.load(path))


    def load(self, k, i):
        return self.loaders[k](i)


    def load_datum(self, i):
        d = dict()
        for k in self._keys:
            d[k] = self.load(k, i)
        return d


class DataProcessing(object):
    """Process raw data from index. Resize, create heatmaps, stickman and
    crops."""
    def __init__(self, size, box_factor):
        self.size = [size, size]
        self.box_factor = box_factor


    def resize(self, x):
        zoom = [self.size[0] / x.shape[0],
                self.size[1] / x.shape[1]]
        if len(x.shape) == 3:
            zoom.append(1.0)
        x = scipy.ndimage.zoom(x, zoom)
        return x


    def __call__(self, image, keypoints, mask, **kwargs):
        assert len(image.shape) == 3
        assert image.shape[0] == image.shape[1]

        image = self.resize(image)

        mask = self.resize(mask)
        mask = np.expand_dims(mask, -1)

        # TODO confidences should be part of data
        confidences = np.ones(keypoints.shape[0], dtype = "float32")
        confidences[np.any(keypoints < 0.0, axis = 1)] = 0.0
        
        heatmaps = make_heatmaps(keypoints, confidences)
        heatmaps = self.resize(heatmaps)

        stickman = make_stickman(Index.joint_order, keypoints, confidences)
        stickman = self.resize(stickman)

        crops = make_crops(image, keypoints,
                Index.joint_order, self.box_factor)

        out = dict(kwargs)
        out["image"] = image
        out["keypoints"] = keypoints
        out["mask"] = mask
        out["heatmaps"] = heatmaps
        out["stickman"] = stickman
        out["crops"] = crops
        return out


class ProcessedIndex(Index):
    """An index with data processing applied."""
    def __init__(self, path, process):
        super().__init__(path)
        self.process = process


    def load_datum(self, i):
        d = super().load_datum(i)
        d.update(self.process(**d))
        return d


class Dataset(DatasetMixin):
    """Dataset of single examples."""
    def __init__(self, index):
        '''Single samples from index.

        Args:
            index (Index): Index to get data from.
        '''
        self.index = index
        self.n = len(self.index)

    def __len__(self):
        return self.n

    def get_example(self, i):
        '''Given an index i, returns a example.
        '''

        return self.index.load_datum(i)


class SeqDataset(Dataset):
    """Dataset of sequences."""
    def __init__(self, index, timespan=3):
        '''Samples corresponding keypoints and images from a video dataset.

        Args:
            index (Index): Index to get data from.
            size (int): Edge length of a returned image/stickman
            timespan (int): Sequence length extracted from video.
        '''
        super().__init__(index)
        self.timespan = timespan
        # only select the timespan'th frame to be able to return sequence
        self.valid_indices = [
                i for i in range(len(self.index)) if
                self.index["fid"][i] >= self.timespan - 1]
        self.n = len(self.valid_indices)


    def get_example(self, i):
        '''Given an index i, returns a sequence of examples.
        '''
        idx = self.valid_indices[i]
        # use corresponding frame and timespan - 1 preceding ones
        seq_indices = list(range(idx - self.timespan + 1, idx + 1))
        # load data
        seq_data = [self.index.load_datum(i) for i in seq_indices]
        # list to dict
        example = dict()
        for s in range(len(seq_data)):
            for k in seq_data[s]:
                example[k + "_{}".format(s)] = seq_data[s][k]
        return example


class Iterator(MultiprocessIterator):
    """Iterator that converts a list of dicts into a dict of lists."""

    def _lod2dol(self, lod):
        return {k: np.stack([d[k] for d in lod], 0) for k in lod[0]}

    def __next__(self):
        return self._lod2dol(super(Iterator, self).__next__())

    @property
    def n(self):
        return len(self.dataset)

    def __len__(self):
        return math.ceil(self.n / self.batch_size)


class FakeDataSet(object):
    def __init__(self, size=100):
        self.n = size
        self.current = 0

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        bs = 16
        imsize = 128
        blocksize = 64
        if self.current < self.n:
            arr = lambda size: np.random.random(size=size)
            feeds = {
                    'original_image': arr([bs, imsize, imsize, 3]),
                    'original_mask': arr([bs, imsize, imsize, 1]),
                    'stickman': arr([bs, imsize, imsize, 3]),
                    'image_blocks': arr([bs, blocksize, blocksize, 24]),
                    'stickman_blocks': arr([bs, blocksize, blocksize, 24])
                    }
            self.current += 1

            return feeds
        else:
            raise StopIteration

    def __len__(self):
        return self.n


def make_batches(dataset, batch_size, shuffle):
    batches = Iterator(dataset,
                       repeat=True,
                       batch_size=batch_size,
                       n_processes=16,
                       shuffle=shuffle)
    return batches


def get_batches(path, size, box_factor, batch_size, shuffle=True):
    process = DataProcessing(size, box_factor)
    index = ProcessedIndex(path, process)
    dataset = Dataset(index)
    return make_batches(dataset, batch_size, shuffle)


def example(path):
    process = DataProcessing(128, 2)
    index = ProcessedIndex(path, process)
    pid, vid = 0, 0
    for frame in range(3):
        print("====== ({}, {}, {}) ======".format(pid, vid, frame))
        idx = index.get_index(pid,vid,frame)
        datum = index.load_datum(idx)
        assert index.get_index(datum["pid"], datum["vid"], datum["fid"]) == idx
        datum["image_relpath"] = index["image"][idx]
        for k,v in sorted(datum.items()):
            if hasattr(v, "shape"):
                print("{: <16}: {}".format(k, v.shape))
            else:
                print("{: <16}: {}".format(k, v))


def example_seq(path):
    process = DataProcessing(128, 2)
    index = ProcessedIndex(path, process)
    data = SeqDataset(index, 3)
    for i in range(2):
        print("====== {} ======".format(i))
        datum = data[i]
        for k,v in sorted(datum.items()):
            if hasattr(v, "shape"):
                print("{: <16}: {}".format(k, v.shape))
            else:
                print("{: <16}: {}".format(k, v))


if __name__ == "__main__":
    import sys, yaml
    import argparse

    P = argparse.ArgumentParser()
    P.add_argument('--config', type=str, help='Path to config', default=None)
    P.add_argument('--index', type=str, help='Path to index pickle', default=None)

    opt = P.parse_args()

    if opt.config is not None:
        with open(opt.config) as f:
            config = yaml.load(f)
        print(config)
        batches = get_batches(config)
    elif opt.index is not None:
        print("Single examples")
        example(opt.index)
        print("Sequence examples")
        example_seq(opt.index)
        exit(0)
    else:
        batches = get_batches(None, fake_data=True)

    for i in range(2):
        batch = next(batches)
        print(list((k,batch[k].shape) for k in batch))
