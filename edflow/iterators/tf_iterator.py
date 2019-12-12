import tensorflow as tf
from edflow.iterators.model_iterator import PyHookedModelIterator


class TFHookedModelIterator(PyHookedModelIterator):
    def make_feeds(self, batch):
        feeds = {
            pl: batch[name] for name, pl in self.model.inputs.items() if name in batch
        }
        return feeds

    def run(self, fetches, feed_dict):
        get_global_step = fetches.pop("global_step")
        results = self.session.run(fetches, feed_dict=feed_dict)
        results["global_step"] = get_global_step()
        return results

    def iterate(self, batch_iterator, validation_batch_iterator=None):
        with self.session.as_default():
            super().iterate(batch_iterator)

    @property
    def session(self):
        # session that is initialized the first time it is needed
        if hasattr(self, "_session"):
            return self._session
        sess_config = tf.ConfigProto()
        if self.config.get("nogpu", False):
            self.logger.info("Hiding GPUs.")
            sess_config.device_count["GPU"] = 0
        sess_config.gpu_options.allow_growth = self.config.get("gpu_mem_growth", False)
        gpu_mem_fraction = self.config.get("gpu_mem_fraction", None)
        if gpu_mem_fraction is not None:
            self.logger.info("Setting GPU MEM Fraction to {}".format(gpu_mem_fraction))
            sess_config.gpu_options.per_process_gpu_memory_fraction = gpu_mem_fraction
        self._session = tf.Session(config=sess_config)
        return self._session
