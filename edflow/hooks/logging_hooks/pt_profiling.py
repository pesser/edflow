import os
import torch
from datetime import datetime

from edflow.hooks.hook import Hook
from edflow.project_manager import ProjectManager as P
from edflow.custom_logging import get_logger


class PyProfilingHook(Hook):
    '''Allows to profile your pytorch code!

    Best of all: It allows you to create chrome traces, which can be view
    interactively in you browser:

    1. Add the hook: 

    ``` python
    class MyIterator(TemplateIterator):
        def __init__(...):
            ...
            self.hooks += [ProfilingHook()]
    ```

    2. Run edflow as usual
    3. Open Chrome and enter ``chrome://tracing``. Use the ``Load`` button to
    open the generated trace, which can be found at ``<Project
    root>/train/profiling/<date>.ctrace``

    '''

    def __init__(self, use_cuda=True, record_shapes=False, ):
        self.logger = get_logger(self)

        self.log_root = os.path.join(P.root, 'profiling')
        os.makedirs(self.log_root, exist_ok=True)

        self.profiler = torch.autograd.profiler.profile(
            use_cuda=use_cuda,
            record_shapes=record_shapes
        )

        self.profiler.__enter__()

    def save_trace(self):
        self.profiler.__exit__(None, None, None)

        now = datetime.now()
        time = date_time = now.strftime("%Y-%m-%dT%H-%M-%S")
        save_path = os.path.join(self.log_root, f'{time}.ctrace')
        self.profiler.export_chrome_trace(save_path)

        self.logger.info(f'Saved profiling trace to {save_path}')

    def at_exception(self, exception):
        self.save_trace()

    def after_epoch(self, epoch):
        self.save_trace()
