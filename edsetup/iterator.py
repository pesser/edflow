from edflow.iterators.template_iterator import TemplateIterator


class Iterator(TemplateIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save(self, checkpoint_path):
        pass

    def restore(self, checkpoint_path):
        pass

    def step_op(self, model, **kwargs):
        inputs, labels = kwargs["inputs"], kwargs["labels"]

        outputs = model(inputs)

        def train_op():
            pass

        def log_op():
            return {
                "inputs": inputs,
                "labels": labels,
            }

        def eval_op():
            return {
                "outputs": outputs,
            }

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}
