class Model(object):
    """
    Clean model skeleton for initialization.
    """

    def __init__(self, config):
        self.config = config

    def __call__(self, x):
        return x
