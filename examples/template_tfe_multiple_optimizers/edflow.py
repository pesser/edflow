import functools
import tensorflow as tf

tf.enable_eager_execution()
import tensorflow.keras as tfk
import numpy as np
from edflow import TemplateIterator, get_logger


class FullLatentDistribution(object):
    # TODO: write some comment on where this comes from
    def __init__(self, parameters, dim, stochastic=True):
        self.parameters = parameters
        self.dim = dim
        self.stochastic = stochastic

        ps = self.parameters.shape.as_list()
        if len(ps) != 2:
            self.expand_dims = True
            self.parameters = tf.reshape(self.parameters, (ps[0], ps[3]))
            ps = self.parameters.shape.as_list()
        else:
            self.expand_dims = False

        assert len(ps) == 2
        self.batch_size = ps[0]

        event_dim = self.dim
        n_L_parameters = (event_dim * (event_dim + 1)) // 2

        size_splits = [event_dim, n_L_parameters]

        self.mean, self.L = tf.split(self.parameters, size_splits, axis=1)
        # L is Cholesky parameterization
        self.L = tf.contrib.distributions.fill_triangular(self.L)
        # make sure diagonal entries are positive by parameterizing them
        # logarithmically
        diag_L = tf.linalg.diag_part(self.L)
        self.log_diag_L = diag_L  # keep for later computation of logdet
        diag_L = tf.exp(diag_L)
        # scale down then set diags
        row_weights = np.array([np.sqrt(i + 1) for i in range(event_dim)])
        row_weights = np.reshape(row_weights, [1, event_dim, 1])
        self.L = self.L / row_weights
        self.L = tf.linalg.set_diag(self.L, diag_L)
        self.Sigma = tf.matmul(self.L, self.L, transpose_b=True)  # L times L^t

        ms = self.mean.shape.as_list()
        self.event_axes = list(range(1, len(ms)))
        self.event_shape = ms[1:]
        assert len(self.event_shape) == 1, self.event_shape

    @staticmethod
    def n_parameters(dim):
        return dim + (dim * (dim + 1)) // 2

    def sample(self, noise_level=1.0):
        if not self.stochastic:
            out = self.mean
        else:
            eps = noise_level * tf.random_normal([self.batch_size, self.dim, 1])
            eps = tf.matmul(self.L, eps)
            eps = tf.squeeze(eps, axis=-1)
            out = self.mean + eps
        if self.expand_dims:
            out = tf.expand_dims(out, axis=1)
            out = tf.expand_dims(out, axis=1)
        return out

    def kl(self, other=None):
        if other is not None:
            raise NotImplemented("Only KL to standard normal is implemented.")

        delta = tf.square(self.mean)
        diag_covar = tf.reduce_sum(tf.square(self.L), axis=2)
        logdet = 2.0 * self.log_diag_L

        kl = 0.5 * tf.reduce_sum(
            diag_covar - 1.0 + delta - logdet, axis=self.event_axes
        )
        kl = tf.reduce_mean(kl)
        return kl


class Model(tfk.Model):
    def __init__(self, config):
        super().__init__()
        self.z_dim = config["z_dim"]
        self.n_z_params = FullLatentDistribution.n_parameters(self.z_dim)

        self.lr = config["lr"]

        self.encode = tfk.Sequential(
            [
                tfk.layers.Dense(
                    1000,
                    kernel_initializer="he_uniform",
                    bias_initializer="random_uniform",
                ),
                tfk.layers.LeakyReLU(0.1),
                tfk.layers.Dense(
                    500,
                    kernel_initializer="he_uniform",
                    bias_initializer="random_uniform",
                ),
                tfk.layers.LeakyReLU(0.1),
                tfk.layers.Dense(
                    300,
                    kernel_initializer="he_uniform",
                    bias_initializer="random_uniform",
                ),
                tfk.layers.LeakyReLU(0.1),
                tfk.layers.Dense(
                    self.n_z_params,
                    kernel_initializer="he_uniform",
                    bias_initializer="random_uniform",
                ),
            ]
        )

        self.decode = tfk.Sequential(
            [
                tfk.layers.Dense(300, kernel_initializer="he_uniform"),
                tfk.layers.LeakyReLU(0.1),
                tfk.layers.Dense(500, kernel_initializer="he_uniform"),
                tfk.layers.LeakyReLU(0.1),
                tfk.layers.Dense(1000, kernel_initializer="he_uniform"),
                tfk.layers.LeakyReLU(0.1),
                tfk.layers.Dense(784, kernel_initializer="he_uniform"),
                tfk.layers.Activation(tf.nn.tanh),
            ]
        )

        input_shape = (config["batch_size"], 28 ** 2)
        self.build(input_shape)

        self.submodels = {"decoder": self.decode, "encoder": self.encode}

    def call(self, x):
        x = tf.reshape(x, (-1, 28 ** 2))
        posterior_params = self.encode(x)
        posterior_distr = FullLatentDistribution(posterior_params, self.z_dim)
        posterior_sample = posterior_distr.sample()

        rec = self.decode(posterior_sample)
        rec = tf.reshape(rec, (-1, 28, 28, 1))
        output = {"x": x, "posterior_distr": posterior_distr, "rec": rec}
        return output


class Iterator(TemplateIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # loss and optimizers

        self.optimizers = {
            submodel_name: tf.compat.v1.train.AdamOptimizer(learning_rate=self.model.lr)
            for submodel_name, submodel in self.model.submodels.items()
        }

        # to save and restore
        self.tfcheckpoint = tf.train.Checkpoint(model=self.model, **self.optimizers)

    def save(self, checkpoint_path):
        self.tfcheckpoint.write(checkpoint_path)

    def restore(self, checkpoint_path):
        self.tfcheckpoint.restore(checkpoint_path)

    def step_op(self, model, **kwargs):
        # get inputs
        losses = {}
        inputs = kwargs["image"]

        # compute loss
        with tf.GradientTape(persistent=True) as tape:
            outputs = model(inputs)
            loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(inputs - outputs["rec"]), axis=(1, 2, 3))
            )
            loss_kl = outputs["posterior_distr"].sample()
            losses["encoder"] = loss + loss_kl
            losses["decoder"] = loss

        def train_op():
            for loss_name, loss in losses.items():
                optimizer = self.optimizers[loss_name]
                submodel = self.model.submodels[loss_name]
                params = submodel.trainable_variables
                grads = tape.gradient(loss, params)
                optimizer.apply_gradients(zip(grads, params))

        image_logs = {"rec": np.array(outputs["rec"]), "x": np.array(inputs)}
        scalar_logs = {"loss_rec": loss, "loss_kl": loss_kl}

        def log_op():
            return {
                "images": image_logs,
                "scalars": scalar_logs,
            }

        def eval_op():
            eval_outputs = {}
            eval_outputs.update(image_logs)
            return eval_outputs

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}
