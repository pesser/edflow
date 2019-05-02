'''When creating, training and evaluating models and/or dataset, the core
object used is the config. It is a dict like object, which contains all
information we want to use e.g. to compare different experiments.
Some of these information will be used change more than others.

A common workflow is to put all frequently tweaked parameters into the ``yaml``
files, which are loaded by edflow, while those changed less often (but maybe
need to be changed in the future) are taken from the config by using the
:method:`get`-method, which allows to define a default value in case the key is
not in the config dict. This way, we do not need to put that value in the
``yaml`` file, making it cleaner and more readable.

Using normal python ``dict``s, there would be one major drawback of this
worklfow, which is that the :method:`get` does not add the default value to the
dictionary, should the key not exist. This is fixed by our :class:`Config`
object, which does exactly that. This way you can :methd:`get` all the
paramaters, withoud needing to remember all the default values should you want
to get the same key more than once.

To ensure reusability and to document, what parameters have actually been used,
the :class:`Config`-object stores itself each time a new key is added. The
storage location is managed by the :class:`ProjectManager`.
'''

import os
import yaml

from edflow.custom_logging import get_logger
from edflow.util import pp2mkdtable
from edflow.project_manager import ProjectManager as P


class Config(dict):
    '''
    The config object works like a simple ``dict`` with the only
    differences that it adds missing default values when calling
    :method:`get()` and that it stores itself for reuse when being updated.
    '''

    def __init__(self, *args, **kwargs):
        self.logger = get_logger(self)
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, val):
        super().__setitem__(key, val)
        self.store()

    def update(self, other):
        super().update(other)
        self.store()

    def get(self, key, default=None):
        '''Tries to :method:`__getitem__` the :attr:`key` from the
        :attr:`base_dict`. If this is not possible, the :attr:`default` value
        will be added to the :attr:`base_dict`.
        '''
        val = super().get(key, default)
        if key not in self:
            self.logger.debug('Updating and storing config.')
            self[key] = val
            self.store()
        return val

    def store(self):
        '''Stores the config at a location defined by the
        :class:`ProjectManager`'''
        savename = 'config-{}.yaml'.format(P.name)
        savepath = os.path.join(P.config, savename)

        with open(savepath, 'w') as cfile:
            cfile.write(yaml.dump(self, default_flow_style=False))

        self.logger.info('Stored config at {}'.format(savepath))

    def __str__(self):
        return pp2mkdtable(self)


if __name__ == '__main__':
    P('./logs', code_root=None)  # , postfix='ctest')

    pre_c = {'a': 'a', 'b': 1, 'c': 100, 'g': {'a': 100, 'b': 1}}

    C = Config(**pre_c)

    C.get('c', 200)
    C.get('f', 'f')

    print(C)

    C.update({'h': 'h', 'i': 'i', 'g': 'g'})
    print(C)
