import tensorflow as tf
from edflow.iterators.tf_trainer import TFBaseTrainer


from tensorflow.contrib.factorization import KMeans

class TrainModel(object):
    def __init__(self, config):
        self.config = config
        self.define_graph()
        self.variables = tf.global_variables()

    @property
    def inputs(self):
        """
        inputs of model at inference time
        Returns
        -------

        """
        return {"image": self.X, "target": self.Y}

    @property
    def outputs(self):
        """
        outputs of model at inference time
        Returns
        -------

        """
        return {}

    def define_graph(self):
        # inputs
        num_features = self.config.get("num_features")
        num_classes = self.config.get("num_classes")
        self.X = tf.placeholder(tf.float32, shape=[None, num_features], name="X")
        # Labels (for assigning a label to a centroid and testing)
        self.Y = tf.placeholder(tf.float32, shape=[None, num_classes], name="Y")

        self.kmeans = KMeans(inputs=self.X, num_clusters=25, distance_metric='cosine',
                             use_mini_batch=True, mini_batch_steps_per_iteration=self.config["mini_batch_steps_per_iteration"])

        training_graph = self.kmeans.training_graph()
        (all_scores, cluster_idx, scores, cluster_centers_initialized,
        k_means_init_op, k_means_train_op) = training_graph
        self.k_means_vars = {
            "all_scores": all_scores,
            "cluster_idx": cluster_idx,
            "scores": scores,
            "cluster_centers_initialized": cluster_centers_initialized,
            "k_means_init_op": k_means_init_op,
            "k_means_train_op": k_means_train_op
        }



class Trainer(TFBaseTrainer):
    def get_restore_variables(self):
        """ nothing fancy here """
        return super().get_restore_variables()

    def initialize(self, checkpoint_path=None):
        """ in this case, we do not need to initialize anything special """
        # return super().initialize(checkpoint_path)
        pass

    def create_train_op(self):
        self.train_op_list = [self.model.k_means_vars["k_means_train_op"], tf.no_op]
        self.train_op = self.train_op_list[0]
        self.run_once_op = self.make_run_once_op()
        losses = self.make_loss_ops()



    # def make_run_once_op(self):
    #     return self.model.k_means_vars["k_means_init_op"]

    def make_loss_ops(self):
        cluster_idx = self.model.k_means_vars["cluster_idx"][0]
        scores = self.model.k_means_vars["scores"]
        # cluster_idx = cluster_idx  # fix for cluster_idx being a tuple
        avg_distance = tf.reduce_mean(scores[0])

        self.log_ops["avg_distance"] = avg_distance
        # self.log_ops["cluster_centers"] = tf.reduce_sum(self.model.kmeans.cluster)

        # counts = tf.reduce_sum(self.model.Y, axis=0, keep_dims=True)
        # labels_map = tf.math.argmax(counts, axis=1)
        # cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
        # correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(self.model.Y, 1), tf.int32))
        # accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # self.log_ops["accuracy"] = accuracy_op

        # self.log_ops["cluster_idx"] = cluster_idx
        #
        return {}

    def run(self, fetches, feed_dict):
        self.train_op = self.train_op_list[(self.get_global_step() // self.config["mini_batch_steps_per_iteration"]) % 2]
        if self.get_global_step() == 0:
            self.logger.info("Running custom initialization op.")
            init_op = tf.variables_initializer(self.get_init_variables())
            self.session.run(init_op, feed_dict=feed_dict)
            self.session.run(self.model.k_means_vars["k_means_init_op"], feed_dict=feed_dict)
        return super().run(fetches, feed_dict)