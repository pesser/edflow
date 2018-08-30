from edflow.custom_logging import get_logger

class Model(object):
    '''Base class defining all neccessary model methods.'''
    @property
    def inputs(self):
        '''Input placeholders'''
        raise NotImplementedError()

    @property
    def outputs(self):
        '''Output tensors.'''
        raise NotImplementedError()
