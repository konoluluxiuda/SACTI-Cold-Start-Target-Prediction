"""Small CPU TensorFlow donor-weight learner, independent of SACTI."""

import numpy as np


class DonorWeightLearner:
    def __init__(self, settings):
        import tensorflow.compat.v1 as tf

        self.tf = tf
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(settings['seed'])
            self.x = tf.placeholder(tf.float64, [None, 3])
            self.j = tf.placeholder(tf.float64, [None])
            self.labels = tf.sparse_placeholder(tf.float64)
            self.positive = tf.placeholder(tf.int32, [None])
            rng = np.random.RandomState(settings['seed'])
            width = settings['hidden_units']
            self.parameters = [
                tf.Variable(rng.normal(0, 0.1, (3, width)), name='kernel'),
                tf.Variable(np.zeros(width), name='bias'),
                tf.Variable(np.zeros((width, 1)), name='output_kernel')]
            kernel, bias, output = self.parameters
            self.offsets = tf.reshape(tf.matmul(tf.tanh(tf.matmul(self.x, kernel) + bias), output), [-1])
            log_weights = tf.nn.log_softmax(tf.log(self.j) + self.offsets)
            weights = tf.exp(log_weights)
            self.scores = tf.reshape(tf.sparse_tensor_dense_matmul(
                self.labels, weights[:, None], adjoint_a=True), [-1])
            logits = self.scores / settings['temperature']
            # All target candidates and all known query positives enter the loss.
            listwise = tf.reduce_logsumexp(logits) - tf.reduce_mean(tf.gather(logits, self.positive))
            reference = self.j / tf.reduce_sum(self.j)
            kl = tf.reduce_sum(weights * (log_weights - tf.log(reference)))
            penalty = tf.add_n([tf.reduce_sum(p * p) for p in self.parameters])
            self.loss = listwise + settings['kl_weight'] * kl + settings['l2'] * penalty
            optimizer = tf.train.AdamOptimizer(settings['learning_rate'])
            gradients = optimizer.compute_gradients(self.loss, self.parameters)
            clipped, _ = tf.clip_by_global_norm([g for g, _ in gradients], settings['gradient_clip'])
            self.train_op = optimizer.apply_gradients(zip(clipped, self.parameters))
            self.restore_inputs = [tf.placeholder(tf.float64, p.shape) for p in self.parameters]
            self.restore_ops = [p.assign(v) for p, v in zip(self.parameters, self.restore_inputs)]
            initializer = tf.global_variables_initializer()
        self.session = tf.Session(graph=self.graph, config=tf.ConfigProto(
            device_count={'GPU': 0}, intra_op_parallelism_threads=1, inter_op_parallelism_threads=1))
        self.session.run(initializer)

    def values(self):
        return self.session.run(self.parameters)

    def restore(self, values):
        self.session.run(self.restore_ops, dict(zip(self.restore_inputs, values)))

    @staticmethod
    def predict(query, values):
        kernel, bias, output = values
        offsets = (np.tanh(query.features @ kernel + bias) @ output).reshape(-1)
        return query.score(offsets)[0]

    def step(self, query, positives):
        positives = np.asarray(positives, dtype=np.int32)
        if not len(positives) or np.any(positives < 0) or np.any(positives >= query.labels.shape[1]):
            raise ValueError('Training requires valid known positive targets.')
        if not len(query.donors):
            # No parameters influence the fallback; the runner counts this query.
            return None
        labels = query.labels.tocoo()
        feed = {self.x: query.features, self.j: query.jaccard,
                self.labels: self.tf.SparseTensorValue(
                    np.column_stack((labels.row, labels.col)), labels.data, labels.shape),
                self.positive: positives}
        loss, _ = self.session.run([self.loss, self.train_op], feed)
        if not np.isfinite(loss) or not all(np.all(np.isfinite(p)) for p in self.values()):
            raise FloatingPointError('Non-finite donor training update.')
        return float(loss)

    def close(self):
        self.session.close()
