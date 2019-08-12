from edflow.iterators.template_iterator import TemplateIterator


class Iterator(TemplateIterator):
    """
    Clean iterator skeleton for initialization.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save(self, checkpoint_path):
        """
        Function for saving the model at a given state
        Parameters
        ----------
        checkpoint_path: The path where the saved checkpoint should lie.

        """

    def restore(self, checkpoint_path):
        """
        Function for model restoration from a given checkpoint.
        Parameters
        ----------
        checkpoint_path: The path where the checkpoint for restoring lies.

        Returns
        -------
        The restored model from the given checkpoint.
        """
        pass

    def step_op(self, model, **kwargs):
        """
        The main method to be called for training by the iterator. Calculating the loss, optimizer step etc.
        Parameters
        ----------
        model : The given model class.

        Returns
        -------
        A dictionary with `train_op`, `log_op` and `eval_op` keys and their returns as their values.
        """
        inputs, labels = kwargs["inputs"], kwargs["labels"]

        outputs = model(inputs)

        def train_op():
            """Takes care of the training process."""
            pass

        def log_op():
            """
            Takes care of the logging process.
            Returns
            -------
            A dictionary whose values are to be logged.
            """
            return {"inputs": inputs, "labels": labels}

        def eval_op():
            """
            Takes care of the evaluation.
            Returns
            -------
            A dictionary with values to be evaluated.
            """
            return {"outputs": outputs}

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}
