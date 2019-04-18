from edflow.iterators.model_iterator import PyHookedModelIterator
from edflow.hooks.pytorch_hooks import DataPrepHook


class TorchHookedModelIterator(PyHookedModelIterator):
    """
    Iterator class for framework PyTorch, inherited from PyHookedModelIterator.
    Args:
        transform (bool): If the batches are to be transformed to pytorch tensors. Should be true even if your input
                    is already pytorch tensors!
    """

    def __init__(self, *args, transform=True, **kwargs):
        super().__init__(*args, **kwargs)
        # check if the data preparation hook is already supplied.
        check = transform and not any(
            [isinstance(hook, DataPrepHook) for hook in self.hooks]
        )
        if check:
            self.hooks += [DataPrepHook()]
