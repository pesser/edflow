class Model(object):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def __call__(self, x):
        return x
