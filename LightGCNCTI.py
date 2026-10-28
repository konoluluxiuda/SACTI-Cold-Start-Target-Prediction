# coding:utf8
import json
import os
from time import localtime, strftime, time

import numpy as np

from util.gpu import configure_cuda_environment

configure_cuda_environment()

import tensorflow.compat.v1 as tf

tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()

from sklearn.metrics import average_precision_score, roc_auc_score

from base.herbRecommender import herbRecommender
from util.lightgcn import build_normalized_bipartite_adjacency
from util.model_components import EarlyStoppingTracker, resolve_early_stopping


class LightGCNCTI(herbRecommender):
    """Same-input LightGCN adaptation for strict C-P prediction."""

    checkpoint_filename = 'lightgcn_cti_model.ckpt'
    metadata_filename = 'lightgcn_cti.json'
    metadata_label = 'LightGCN'

    def readConfiguration(self):
        super(LightGCNCTI, self).readConfiguration()
        incompatible = []
        for key in (
                'context.interaction',
                'counterfactual.context',
                'context.mask.training',
                'support.router',
                'inductive.context',
                'hyperedge.attention',
                'global.token.attention'):
            if (
                self.config.contains(key)
                and str(self.config[key]).strip().lower()
                in {'1', 'true', 'yes', 'on'}
            ):
                incompatible.append(key)
        if incompatible:
            raise ValueError(
                'LightGCN-CTI is a frozen pair-only baseline; incompatible '
                'settings: %s.' % ', '.join(incompatible)
            )
        self.n_layers = (
            int(self.config['lightgcn.layers'])
            if self.config.contains('lightgcn.layers') else 3
        )
        if self.n_layers < 1:
            raise ValueError('lightgcn.layers must be at least 1.')
        objective = (
            str(self.config['lightgcn.objective']).strip().lower()
            if self.config.contains('lightgcn.objective') else 'bce'
        )
        if objective != 'bce':
            raise ValueError(
                'The frozen same-input LightGCN-CTI baseline requires '
                'lightgcn.objective=bce.'
            )
        self.objective = objective

    def initModel(self):
        if getattr(self.data, 'protocol', 'legacy') != 'strict':
            raise ValueError(
                'LightGCN-CTI requires experiment.protocol=strict.'
            )
        if (
            self.config.contains('pair.decoder')
            and str(self.config['pair.decoder']).strip().lower() != 'dot'
        ):
            raise ValueError('LightGCN-CTI supports only pair.decoder=dot.')

        super(LightGCNCTI, self).initModel()
        graph_edges = [
            (
                self.data.compound[str(compound_id)],
                self.data.protein[str(protein_id)],
            )
            for compound_id, protein_id, label in self.data.cpassociation
            if float(label) > 0
        ]
        self.graph_metadata = build_normalized_bipartite_adjacency(
            self.num_compounds, self.num_proteins, graph_edges
        )
        normalized_adjacency = tf.sparse_reorder(tf.SparseTensor(
            self.graph_metadata['indices'],
            self.graph_metadata['values'],
            self.graph_metadata['shape'],
        ))

        ego_embeddings = tf.concat(
            [self.compound_embeddings, self.protein_embeddings],
            axis=0,
            name='lightgcn_ego_embeddings',
        )
        layer_embeddings = [ego_embeddings]
        propagated_embeddings = ego_embeddings
        for layer in range(1, self.n_layers + 1):
            propagated_embeddings = tf.sparse_tensor_dense_matmul(
                normalized_adjacency,
                propagated_embeddings,
                name='lightgcn_propagation_layer_%d' % layer,
            )
            layer_embeddings.append(propagated_embeddings)

        final_embeddings = tf.reduce_mean(
            tf.stack(layer_embeddings, axis=0),
            axis=0,
            name='lightgcn_uniform_layer_mean',
        )
        self.final_uembedding = final_embeddings[:self.num_compounds]
        self.final_iembedding = final_embeddings[self.num_compounds:]
        self.u_embedding = tf.nn.embedding_lookup(
            self.final_uembedding, self.u_idx
        )
        self.v_embedding = tf.nn.embedding_lookup(
            self.final_iembedding, self.v_idx
        )
        self.logits = tf.reduce_sum(
            self.u_embedding * self.v_embedding,
            axis=1,
            name='lightgcn_pair_logits',
        )
        self.bce_loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.r,
                logits=self.logits,
            )
        )
        self.regularization_loss = (
            self.regU * tf.nn.l2_loss(self.compound_embeddings)
            + self.regI * tf.nn.l2_loss(self.protein_embeddings)
        )
        self.loss = self.bce_loss + self.regularization_loss
        self.train_operation = tf.train.AdamOptimizer(
            self.lRate
        ).minimize(self.loss)

        print(
            'LightGCN-CTI: layers=%d aggregation=uniform_mean '
            'objective=%s graph_source=inner_train_positive_C-P.' % (
                self.n_layers, self.objective
            )
        )
        print(
            'LightGCN training graph: edges=%d active_compounds=%d/%d '
            'active_proteins=%d/%d hash=%s.' % (
                self.graph_metadata['edge_count'],
                self.graph_metadata['active_compounds'],
                self.num_compounds,
                self.graph_metadata['active_proteins'],
                self.num_proteins,
                self.graph_metadata['edge_sha256'][:12],
            )
        )

    @staticmethod
    def _format_duration(seconds):
        seconds = max(0, int(seconds))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        if hours:
            return '%dh%02dm%02ds' % (hours, minutes, secs)
        if minutes:
            return '%dm%02ds' % (minutes, secs)
        return '%ds' % secs

    def fetchModelState(self):
        compound, protein = self.sess.run([
            self.final_uembedding,
            self.final_iembedding,
        ])
        return {
            'compound': compound,
            'protein': protein,
        }

    def evaluateValidation(self, state, metric):
        if not self.validationData:
            raise ValueError(
                'Early stopping is enabled but no inner validation records '
                'were provided.'
            )
        compound_indices = []
        protein_indices = []
        labels = []
        for compound_id, protein_id, label in self.validationData:
            compound_indices.append(self.data.compound[str(compound_id)])
            protein_indices.append(self.data.protein[str(protein_id)])
            labels.append(1 if float(label) > 0 else 0)
        labels = np.asarray(labels, dtype=np.int32)
        if len(np.unique(labels)) < 2:
            raise ValueError(
                'Inner validation must contain both positive and negative '
                'records.'
            )
        logits = np.sum(
            state['compound'][compound_indices]
            * state['protein'][protein_indices],
            axis=1,
        )
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
        if metric == 'aupr':
            return float(average_precision_score(labels, scores))
        if metric == 'auc':
            return float(roc_auc_score(labels, scores))
        raise ValueError('Unsupported validation metric: %s' % metric)

    def buildTrainingMetadata(self):
        graph = {
            key: value
            for key, value in self.graph_metadata.items()
            if key not in {'indices', 'values'}
        }
        graph['shape'] = list(graph['shape'])
        return {
            'model_role': 'LightGCN-CTI same-input BCE adaptation',
            'layers': self.n_layers,
            'aggregation': 'uniform_mean',
            'objective': self.objective,
            'graph_source': 'strict_inner_train_positive_C-P',
            'graph': graph,
        }

    def trainModel(self):
        self.sess.run(tf.global_variables_initializer())
        early_stopping = resolve_early_stopping(self.config)
        if early_stopping['enabled'] and not self.validationData:
            raise ValueError(
                'early.stopping=True requires a non-empty inner validation '
                'split.'
            )
        tracker = None
        if early_stopping['enabled']:
            tracker = EarlyStoppingTracker(
                early_stopping['patience'],
                early_stopping['min_delta'],
            )
            print(
                'Early stopping: metric=%s interval=%d patience=%d '
                'min_delta=%g validation_pairs=%d' % (
                    early_stopping['metric'].upper(),
                    early_stopping['interval'],
                    early_stopping['patience'],
                    early_stopping['min_delta'],
                    len(self.validationData),
                )
            )

        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        model_dir = os.path.join('./saved_model', current_time)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, self.checkpoint_filename)
        saver = tf.train.Saver(max_to_keep=1)

        total_batches = int(np.ceil(float(self.train_size) / self.batch_size))
        total_steps = max(1, self.maxEpoch * total_batches)
        train_start = time()
        epochs_completed = 0
        for epoch in range(self.maxEpoch):
            epoch_start = time()
            for batch_number, batch in enumerate(
                    self.next_batch_pairwise(), start=1):
                batch_start = time()
                compound_indices, protein_indices, labels = batch
                _, loss_value = self.sess.run(
                    [self.train_operation, self.loss],
                    feed_dict={
                        self.u_idx: compound_indices,
                        self.v_idx: protein_indices,
                        self.r: labels,
                    },
                )
                if not np.isfinite(loss_value):
                    raise ValueError(
                        'Training loss became non-finite at epoch %d batch '
                        '%d: %s' % (
                            epoch + 1, batch_number, loss_value
                        )
                    )
                elapsed = time() - train_start
                step = min(
                    epoch * total_batches + batch_number,
                    total_steps,
                )
                eta = elapsed / step * (total_steps - step)
                print(
                    'training: %d/%d batch %d/%d loss: %s batch_time: %s '
                    'elapsed: %s eta: %s' % (
                        epoch + 1,
                        self.maxEpoch,
                        batch_number,
                        total_batches,
                        loss_value,
                        self._format_duration(time() - batch_start),
                        self._format_duration(elapsed),
                        self._format_duration(eta),
                    )
                )
            epochs_completed = epoch + 1
            print(
                'epoch %d/%d finished in %s' % (
                    epoch + 1,
                    self.maxEpoch,
                    self._format_duration(time() - epoch_start),
                )
            )

            should_validate = early_stopping['enabled'] and (
                (epoch + 1) % early_stopping['interval'] == 0
                or epoch + 1 == self.maxEpoch
            )
            if should_validate:
                validation_value = self.evaluateValidation(
                    self.fetchModelState(),
                    early_stopping['metric'],
                )
                improved, should_stop = tracker.update(
                    validation_value, epoch + 1
                )
                if improved:
                    saver.save(self.sess, model_path)
                print(
                    'validation: epoch %d %s=%.6f best=%.6f '
                    'best_epoch=%d stale=%d/%d%s' % (
                        epoch + 1,
                        early_stopping['metric'].upper(),
                        validation_value,
                        tracker.best_value,
                        tracker.best_epoch,
                        tracker.stale_checks,
                        tracker.patience,
                        ' improved' if improved else '',
                    )
                )
                if should_stop:
                    print(
                        'Early stopping triggered at epoch %d.' %
                        (epoch + 1)
                    )
                    break

        if early_stopping['enabled']:
            saver.restore(self.sess, model_path)
            self.early_stopping_summary = {
                'best_epoch': tracker.best_epoch,
                'best_value': tracker.best_value,
                'epochs_completed': epochs_completed,
                'metric': early_stopping['metric'],
            }
            print(
                'Restored best validation checkpoint: epoch %d %s=%.6f' % (
                    tracker.best_epoch,
                    early_stopping['metric'].upper(),
                    tracker.best_value,
                )
            )
        else:
            saver.save(self.sess, model_path)
            self.early_stopping_summary = None

        state = self.fetchModelState()
        self.u = state['compound']
        self.i = state['protein']
        metadata = self.buildTrainingMetadata()
        metadata['early_stopping'] = self.early_stopping_summary
        metadata_path = os.path.join(model_dir, self.metadata_filename)
        with open(metadata_path, 'w', encoding='utf-8') as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
        print('%s metadata: %s' % (self.metadata_label, metadata_path))
        print('模型权重保存成功: %s' % model_path)

    def predictForPairs(self, compound_indices, protein_indices):
        compound_indices = np.asarray(compound_indices, dtype=np.int64)
        protein_indices = np.asarray(protein_indices, dtype=np.int64)
        return np.sum(
            self.u[compound_indices] * self.i[protein_indices],
            axis=1,
        )

    def predictForRanking(self):
        return self.u.dot(self.i.transpose())
