from edflow.hooks.hook import Hook


class MetricHook(Hook):
    """Applies a set of given metrics to the calculated data."""

    def __init__(self, metrics, save_root, consider_only_first=None):
        """Args:
            metrics (list): List of ``MetricTuple``s of the form
                ``(input names, output names, metric, name)``.
                - ``input names`` are the keys corresponding to the feeds of
                    interest, e.g. an original image.
                - ``output names`` are the keys corresponding to the values
                    in the results dict.
                - ``metric`` is a ``Callable`` that accepts all inputs and
                    outputs keys as keyword arguments
                - ``name`` is a
                If nested feeds or results are expected the names can be
                passed as "path" like ``'key1_key2'`` returning
                ``dict[key1][key2]``.
            save_root (str): Path to where the results are stored.
            consider_only_first (int): Metric is only evaluated on the first
                `consider_only_first` examples.
        """

        self.metrics = metrics

        self.root = save_root
        self.logger = get_logger(self, "latest_eval")

        self.max_step = consider_only_first

        self.storage_dict = {}
        self.metric_results = {}
        for m in metrics:
            test_valid_metrictuple(m)

        self.tb_saver = tf.summary.FileWriter(self.root)

    def before_epoch(self, epoch):
        self.count = 0
        for m in self.metrics:
            self.metric_results[m.name] = []

    def before_step(self, step, fetches, feeds, batch):
        if self.max_step is not None and self.count >= self.max_step:
            return

        for in_names, out_names, metric, m_name in self.metrics:
            self.storage_dict[m_name] = {}
            for kwargs_name, name in in_names.items():
                val = retrieve(batch, name)
                self.storage_dict[m_name][kwargs_name] = val

    def after_step(self, step, results):
        if self.max_step is not None and self.count >= self.max_step:
            return

        for in_names, out_names, metric, m_name in self.metrics:
            for kwargs_name, name in out_names.items():
                val = retrieve(results, name)
                self.storage_dict[m_name][kwargs_name] = val
            m_res = metric(**self.storage_dict[m_name])
            self.metric_results[m_name] += [m_res]

        self.global_step = results["global_step"]
        self.count += 1

    def after_epoch(self, epoch):
        self.logger.info("Metrics at epoch {}:".format(epoch))

        mean_results = {}
        for name, result in self.metric_results.items():
            results = np.concatenate(result)
            mean = np.mean(results, axis=0)
            var = np.std(results, axis=0)
            mean_results[name] = np.array([mean, var])
            self.logger.info("{}: {} +- {}".format(name, mean, var))

            summary = tf.Summary()
            summary_mean = mean if len(mean.shape) == 0 else mean[0]
            summary.value.add(tag=name, simple_value=summary_mean)
            self.tb_saver.add_summary(summary, self.global_step)
            self.tb_saver.flush()

        name = "{:0>6d}_metrics".format(self.global_step)
        name = os.path.join(self.root, name)
        np.savez_compressed(name, **mean_results)
