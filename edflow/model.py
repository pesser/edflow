import tensorflow as tf


class Model(object):
    """Base class defining all neccessary model methods."""

    def __init__(self, name):
        self.model_name = name

    @property
    def inputs(self):
        """Input placeholders"""
        raise NotImplementedError()

    @property
    def outputs(self):
        """Output tensors."""
        raise NotImplementedError()

    @property
    def variables(self):
        """Variables."""
        return [v for v in tf.global_variables() if self.model_name in v.name]
