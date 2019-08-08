class Model(object):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x):
        return x