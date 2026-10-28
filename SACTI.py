# coding:utf8
"""Support-Aware Context Transfer for compound-target interaction prediction.

SACTI is intentionally implemented as an independent recommender.  It shares
only the repository's data/evaluation interface with the legacy models and
does not import or inherit the HDCTI implementation.
"""

import json
import os
from time import localtime, strftime, time

import numpy as np
import scipy.sparse as sp

from util.gpu import configure_cuda_environment

configure_cuda_environment()

import tensorflow.compat.v1 as tf

tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()

from sklearn.metrics import average_precision_score, roc_auc_score

from base.herbRecommender import herbRecommender
from util.dual_granularity_transfer import (
    build_loco_jaccard_transfer,
    loco_jaccard_pair_residuals,
)
from util.herb_prototype_transfer import (
    build_support_calibrated_herb_prototypes,
    support_calibrated_herb_prototype_scores,
)
from util.model_components import EarlyStoppingTracker, resolve_early_stopping
from util.pseudo_cold_training import query_blocks, masked_transfer_state
from util.side_relation_reconstruction import (
    leave_one_herb_out_memberships, HerbReconstructionSampler)


def _config_bool(config, key, default):
    if not config.contains(key):
        return default
    value = str(config[key]).strip().lower()
    if value in {'1', 'true', 'yes', 'on'}:
        return True
    if value in {'0', 'false', 'no', 'off'}:
        return False
    raise ValueError('Invalid boolean value for %s: %s' % (key, config[key]))


def _binary_matrix(rows, columns, pairs):
    row_indices = []
    column_indices = []
    for row_index, column_index in pairs:
        row_indices.append(int(row_index))
        column_indices.append(int(column_index))
    values = np.ones(len(row_indices), dtype=np.float32)
    matrix = sp.coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(rows, columns),
        dtype=np.float32,
    ).tocsr()
    if matrix.nnz:
        matrix.data[:] = 1.0
        matrix.eliminate_zeros()
    return matrix


def _row_normalized_sparse(matrix):
    matrix = sp.csr_matrix(matrix, dtype=np.float32)
    degrees = np.asarray(matrix.sum(axis=1), dtype=np.float32).reshape(-1)
    inverse = np.divide(
        1.0,
        degrees,
        out=np.zeros_like(degrees),
        where=degrees > 0,
    )
    return sp.diags(inverse, dtype=np.float32).dot(matrix).tocoo()


def _tf_sparse(matrix, name):
    matrix = matrix.tocoo()
    indices = np.asarray([matrix.row, matrix.col], dtype=np.int64).T
    tensor = tf.SparseTensor(
        indices=indices,
        values=matrix.data.astype(np.float32),
        dense_shape=matrix.shape,
    )
    return tf.sparse_reorder(tensor, name=name)


class SACTI(herbRecommender):
    """Independent support-aware context-transfer model.

    The model contains four native score components:

    1. a supported compound-ID expert;
    2. candidate-conditioned herb-context interaction;
    3. fold-local leave-one-compound-out herb prototypes;
    4. fold-local H-C Jaccard label propagation.

    H-C and P-D contexts are encoded by sparse membership averaging followed
    by residual MLPs.  No C-P PageRank, self-gating, dense all-node attention,
    or node-dimension attention is present.
    """

    checkpoint_filename = 'sacti_model.ckpt'
    metadata_filename = 'sacti.json'

    def readConfiguration(self):
        super(SACTI, self).readConfiguration()
        if getattr(self.data, 'protocol', 'legacy') != 'strict':
            raise ValueError('SACTI requires experiment.protocol=strict.')
        if self.config.contains('encoder.profile'):
            raise ValueError(
                'SACTI has an independent encoder and does not accept '
                'encoder.profile.'
            )
        forbidden = []
        for key in (
                'hyperedge.attention',
                'global.token.attention',
                'hplga.enabled',
                'support.router',
                'support.experts',
                'support.state.routing'):
            if _config_bool(self.config, key, False):
                forbidden.append(key)
        if forbidden:
            raise ValueError(
                'SACTI does not use legacy/profile modules: %s.' %
                ', '.join(forbidden)
            )

        self.context_layers = (
            int(self.config['sacti.context.layers'])
            if self.config.contains('sacti.context.layers') else 2
        )
        self.prototype_enabled = _config_bool(
            self.config, 'sacti.prototype', True
        )
        self.jaccard_enabled = _config_bool(
            self.config, 'sacti.jaccard', True
        )
        self.support_id_gate_enabled = _config_bool(
            self.config, 'sacti.support.id.gate', True
        )
        self.context_expert_enabled = _config_bool(
            self.config, 'sacti.context.expert', True
        )
        self.hc_reconstruction_weight = float(
            self.config['sacti.hc.reconstruction.weight']
            if self.config.contains('sacti.hc.reconstruction.weight') else 0.0
        )
        if not np.isfinite(self.hc_reconstruction_weight) or self.hc_reconstruction_weight < 0:
            raise ValueError('H-C reconstruction weight must be finite and nonnegative.')
        if self.hc_reconstruction_weight > 0 and not self.context_expert_enabled:
            raise ValueError('H-C reconstruction requires the context scoring expert.')
        self.hc_reconstruction_samples = int(
            self.config['sacti.hc.reconstruction.samples']
            if self.config.contains('sacti.hc.reconstruction.samples') else 32)
        self.hc_reconstruction_seed = int(
            self.config['sacti.hc.reconstruction.seed']
            if self.config.contains('sacti.hc.reconstruction.seed') else 152026)
        if self.hc_reconstruction_samples < 1:
            raise ValueError('H-C reconstruction sample count must be positive.')
        self.training_mask_id = _config_bool(
            self.config, 'sacti.training.mask.id', False
        )
        self.exclude_query_prior = _config_bool(
            self.config, 'sacti.transfer.exclude.query.prior', False
        )
        self.raw_transfer_scores = _config_bool(
            self.config, 'sacti.transfer.raw.scores', False
        )
        if self.raw_transfer_scores:
            self.exclude_query_prior = True
        self.training_mask_evidence = _config_bool(
            self.config, 'sacti.training.mask.evidence', False
        )
        self.training_block_count = int(
            self.config['sacti.training.blocks']
            if self.config.contains('sacti.training.blocks') else 5
        )
        self.training_block_seed = int(
            self.config['sacti.training.block.seed']
            if self.config.contains('sacti.training.block.seed') else 152026
        )
        if self.training_mask_evidence and not self.training_mask_id:
            raise ValueError('Training evidence masking requires training ID masking.')
        if self.training_mask_id and not self.support_id_gate_enabled:
            raise ValueError('Pseudo-cold training requires the support ID gate.')
        if self.training_mask_id and self.training_block_count < 2:
            raise ValueError('sacti.training.blocks must be at least two.')
        self.prototype_prior = (
            float(self.config['sacti.prototype.prior'])
            if self.config.contains('sacti.prototype.prior') else 1.0
        )
        self.weight_reg = (
            float(self.config['weight.reg'])
            if self.config.contains('weight.reg') else 0.01
        )
        self.prediction_batch_size = (
            int(self.config['pair.prediction.batch.size'])
            if self.config.contains('pair.prediction.batch.size') else 65536
        )
        if self.context_layers < 1:
            raise ValueError('sacti.context.layers must be at least 1.')
        if self.prototype_prior <= 0:
            raise ValueError('sacti.prototype.prior must be positive.')
        if self.weight_reg < 0:
            raise ValueError('weight.reg cannot be negative.')
        if self.prediction_batch_size <= 0:
            raise ValueError('pair.prediction.batch.size must be positive.')

    def _build_relation_state(self):
        hc_pairs = [
            (self.data.herb[str(herb)], self.data.compound[str(compound)])
            for herb, compound, *_ in self.data.hcassociation
        ]
        pd_pairs = [
            (self.data.protein[str(protein)], self.data.disease[str(disease)])
            for protein, disease, *_ in self.data.pdassociation
        ]
        cp_pairs = [
            (self.data.compound[str(compound)], self.data.protein[str(protein)])
            for compound, protein, label in self.data.cpassociation
            if float(label) > 0
        ]
        self.hc_matrix = _binary_matrix(
            self.num_herbs, self.num_compounds, hc_pairs
        )
        self.pd_matrix = _binary_matrix(
            self.num_proteins, self.num_diseases, pd_pairs
        )
        self.cp_matrix = _binary_matrix(
            self.num_compounds, self.num_proteins, cp_pairs
        )
        self.compound_herb_membership = _row_normalized_sparse(
            self.hc_matrix.transpose()
        )
        self.protein_disease_membership = _row_normalized_sparse(
            self.pd_matrix
        )
        self.compound_support_degrees = np.asarray(
            self.cp_matrix.getnnz(axis=1), dtype=np.float32
        ).reshape(-1)
        self.compound_context_available = (
            np.asarray(
                self.hc_matrix.getnnz(axis=0), dtype=np.float32
            ).reshape(-1) > 0
        ).astype(np.float32)
        self.protein_context_available = (
            np.asarray(
                self.pd_matrix.getnnz(axis=1), dtype=np.float32
            ).reshape(-1) > 0
        ).astype(np.float32)

        compound_herbs = self.hc_matrix.transpose().tocsr()
        max_herbs = max(
            [
                compound_herbs.indptr[index + 1]
                - compound_herbs.indptr[index]
                for index in range(self.num_compounds)
            ] + [1]
        )
        sentinel = self.num_herbs
        self.compound_herb_indices = np.full(
            (self.num_compounds, max_herbs), sentinel, dtype=np.int32
        )
        self.compound_herb_mask = np.zeros(
            (self.num_compounds, max_herbs), dtype=np.float32
        )
        for compound in range(self.num_compounds):
            start = compound_herbs.indptr[compound]
            stop = compound_herbs.indptr[compound + 1]
            herbs = compound_herbs.indices[start:stop]
            if herbs.size:
                self.compound_herb_indices[compound, :herbs.size] = herbs
                self.compound_herb_mask[compound, :herbs.size] = 1.0

        self.herb_prototypes = build_support_calibrated_herb_prototypes(
            self.hc_matrix, self.cp_matrix
        )
        self.jaccard_transfer = build_loco_jaccard_transfer(
            self.hc_matrix, self.cp_matrix
        )
        self.training_blocks = (
            query_blocks(self.cp_matrix, self.training_block_count,
                         self.training_block_seed)
            if self.training_mask_id else []
        )
        self.training_query_mask = np.zeros(self.num_compounds, dtype=bool)
        self.training_transfer_state = None
        self.training_masked_records = 0
        self.training_epoch_prepared = False
        if self.hc_reconstruction_weight > 0:
            self.hc_reconstruction_sampler = HerbReconstructionSampler(
                self.hc_matrix, np.flatnonzero(self.compound_support_degrees > 0),
                self.hc_reconstruction_seed)

    def _residual_context_encoder(self, inputs, prefix):
        output = tf.nn.l2_normalize(inputs, axis=1)
        initializer = tf.glorot_normal_initializer()
        for layer in range(1, self.context_layers + 1):
            weight = tf.get_variable(
                '%s_residual_weight_%d' % (prefix, layer),
                shape=[self.emb_size, self.emb_size],
                initializer=initializer,
            )
            bias = tf.get_variable(
                '%s_residual_bias_%d' % (prefix, layer),
                shape=[self.emb_size],
                initializer=tf.zeros_initializer(),
            )
            update = tf.nn.relu(
                tf.matmul(output, weight) + bias,
                name='%s_residual_update_%d' % (prefix, layer),
            )
            output = tf.nn.l2_normalize(
                output + update,
                axis=1,
                name='%s_residual_output_%d' % (prefix, layer),
            )
        return output

    def _build_hc_reconstruction(self):
        self.hc_reconstruction_membership = tf.sparse_placeholder(
            tf.float32, [None, self.num_herbs], name='hc_reconstruction_membership')
        self.hc_reconstruction_herbs = tf.placeholder(
            tf.int32, [None], name='hc_reconstruction_herbs')
        self.hc_reconstruction_labels = tf.placeholder(
            tf.float32, [None], name='hc_reconstruction_labels')
        self.hc_reconstruction_valid = tf.placeholder(
            tf.float32, [None], name='hc_reconstruction_valid')
        raw = tf.sparse_tensor_dense_matmul(
            self.hc_reconstruction_membership, self.herb_embeddings)
        with tf.variable_scope('sacti', reuse=True):
            context = self._residual_context_encoder(raw, 'compound_context')
        with tf.variable_scope('sacti_hc_reconstruction'):
            projection = tf.get_variable(
                'projection', [self.emb_size, self.emb_size],
                initializer=tf.glorot_normal_initializer())
        herb = tf.nn.l2_normalize(tf.nn.embedding_lookup(
            self.herb_embeddings, self.hc_reconstruction_herbs), axis=1)
        logits = tf.reduce_sum(tf.matmul(context, projection) * herb, axis=1)
        per_pair = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.hc_reconstruction_labels, logits=logits)
        self.hc_reconstruction_loss = tf.reduce_sum(
            per_pair * self.hc_reconstruction_valid
        ) / tf.maximum(tf.reduce_sum(self.hc_reconstruction_valid), 1.0)

    def hcReconstructionFeed(self, compounds, herbs, labels, masked_herbs=None):
        if self.hc_reconstruction_weight <= 0:
            raise ValueError('H-C reconstruction is disabled.')
        labels = np.asarray(labels, dtype=np.float32)
        if labels.shape != np.asarray(compounds).shape or not np.all(np.isin(labels, [0., 1.])):
            raise ValueError('One binary H-C label is required per pair.')
        rows, valid = leave_one_herb_out_memberships(
            self.hc_matrix, compounds, herbs if masked_herbs is None else masked_herbs)
        # Validate targets independently when context masking is shared by a pair.
        if masked_herbs is not None:
            leave_one_herb_out_memberships(self.hc_matrix, compounds, herbs)
        expected = np.asarray(self.hc_matrix[np.asarray(herbs, dtype=int),
                                            np.asarray(compounds, dtype=int)]).ravel()
        if not np.array_equal(labels, expected):
            raise ValueError('Auxiliary labels must match observed H-C membership.')
        if np.any(np.asarray(rows[np.arange(len(herbs)), np.asarray(herbs, dtype=int)]).ravel()):
            raise ValueError('Reconstruction target must not occur in its input context.')
        coo = rows.tocoo()
        value = tf.SparseTensorValue(
            np.column_stack((coo.row, coo.col)).astype(np.int64),
            coo.data.astype(np.float32), np.asarray(coo.shape, dtype=np.int64))
        return {self.hc_reconstruction_membership: value,
                self.hc_reconstruction_herbs: np.asarray(herbs, dtype=np.int32),
                self.hc_reconstruction_labels: labels,
                self.hc_reconstruction_valid: valid.astype(np.float32)}

    def initModel(self):
        super(SACTI, self).initModel()
        self._build_relation_state()

        initializer = tf.glorot_normal_initializer()
        with tf.variable_scope('sacti'):
            self.herb_embeddings = tf.get_variable(
                'herb_embeddings',
                shape=[self.num_herbs, self.emb_size],
                initializer=initializer,
            )
            self.disease_embeddings = tf.get_variable(
                'disease_embeddings',
                shape=[self.num_diseases, self.emb_size],
                initializer=initializer,
            )
            compound_membership = _tf_sparse(
                self.compound_herb_membership,
                'compound_herb_membership',
            )
            protein_membership = _tf_sparse(
                self.protein_disease_membership,
                'protein_disease_membership',
            )
            compound_context_raw = tf.sparse_tensor_dense_matmul(
                compound_membership,
                self.herb_embeddings,
                name='compound_context_average',
            )
            protein_context_raw = tf.sparse_tensor_dense_matmul(
                protein_membership,
                self.disease_embeddings,
                name='protein_context_average',
            )
            self.final_compound_context = self._residual_context_encoder(
                compound_context_raw, 'compound_context'
            )
            self.final_protein_context = self._residual_context_encoder(
                protein_context_raw, 'protein_context'
            )
            self.protein_context_scale = tf.get_variable(
                'protein_context_scale',
                initializer=np.asarray([1.0], dtype=np.float32),
            )
            self.context_interaction = tf.get_variable(
                'context_interaction',
                shape=[self.emb_size],
                initializer=tf.ones_initializer(),
            )
            self.prototype_scale = tf.get_variable(
                'prototype_scale',
                initializer=np.asarray([0.0], dtype=np.float32),
                trainable=self.prototype_enabled,
            )
            self.jaccard_scale = tf.get_variable(
                'jaccard_scale',
                initializer=np.asarray([0.0], dtype=np.float32),
                trainable=self.jaccard_enabled,
            )

        self.final_compound_id = tf.nn.l2_normalize(
            self.compound_embeddings, axis=1, name='supported_compound_id'
        )
        protein_id = tf.nn.l2_normalize(self.protein_embeddings, axis=1)
        self.final_protein = tf.nn.l2_normalize(
            protein_id
            + self.protein_context_scale[0] * self.final_protein_context,
            axis=1,
            name='protein_id_context_fusion',
        )

        pair_compound_id = tf.nn.embedding_lookup(
            self.final_compound_id, self.u_idx
        )
        pair_compound_context = tf.nn.embedding_lookup(
            self.final_compound_context, self.u_idx
        )
        pair_protein = tf.nn.embedding_lookup(
            self.final_protein, self.v_idx
        )
        pair_support = tf.gather(
            tf.constant(self.compound_support_degrees, dtype=tf.float32),
            self.u_idx,
        )
        pair_context_available = tf.gather(
            tf.constant(self.compound_context_available, dtype=tf.float32),
            self.u_idx,
        )
        self.pair_id_gate = (
            tf.cast(pair_support > 0, tf.float32)
            if self.support_id_gate_enabled else
            tf.ones_like(pair_support, dtype=tf.float32)
        )
        # Add no graph operation in the disabled mode, preserving legacy initialization.
        if self.training_mask_id:
            self.training_id_keep = tf.placeholder_with_default(
                tf.ones_like(pair_support), shape=[None], name='training_id_keep'
            )
            self.pair_id_gate = self.pair_id_gate * self.training_id_keep
        self.pair_context_gate = (
            pair_context_available
            if self.context_expert_enabled else
            tf.zeros_like(pair_context_available, dtype=tf.float32)
        )
        self.prototype_residual = tf.placeholder(
            tf.float32, shape=[None], name='prototype_residual'
        )
        self.jaccard_residual = tf.placeholder(
            tf.float32, shape=[None], name='jaccard_residual'
        )

        id_logits = self.pair_id_gate * tf.reduce_sum(
            pair_compound_id * pair_protein, axis=1
        )
        context_logits = self.pair_context_gate * tf.reduce_sum(
            pair_compound_context
            * pair_protein
            * self.context_interaction,
            axis=1,
        )
        self.logits = tf.identity(
            id_logits
            + context_logits
            + self.prototype_scale[0] * self.prototype_residual
            + self.jaccard_scale[0] * self.jaccard_residual,
            name='support_aware_logits',
        )
        self.bce_loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.r, logits=self.logits
            )
        )
        regularized = [
            self.compound_embeddings,
            self.protein_embeddings,
            self.herb_embeddings,
            self.disease_embeddings,
        ]
        if self.context_expert_enabled:
            regularized.append(self.context_interaction)
        self.regularization_loss = self.weight_reg * tf.add_n([
            tf.nn.l2_loss(variable) for variable in regularized
        ])
        self.loss = self.bce_loss + self.regularization_loss
        if self.hc_reconstruction_weight > 0:
            self._build_hc_reconstruction()
            # Scale the auxiliary mean to the summed main-task BCE batch size.
            self.loss += (self.hc_reconstruction_weight
                          * tf.cast(tf.size(self.r), tf.float32)
                          * self.hc_reconstruction_loss)
        self.train_operation = tf.train.AdamOptimizer(
            self.lRate
        ).minimize(self.loss)

        print(
            'SACTI encoder: independent sparse membership context '
            '(layers=%d); no PageRank/self-gating/full attention.' %
            self.context_layers
        )
        print(
            'SACTI support experts: ID=%s, Hctx-P=%s, prototype=%s, '
            'Jaccard=%s.' % (
                'training-support-only'
                if self.support_id_gate_enabled else 'ungated',
                'active-when-H-C-observed'
                if self.context_expert_enabled else 'off',
                'on' if self.prototype_enabled else 'off',
                'on' if self.jaccard_enabled else 'off',
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

    def transferResiduals(self, compound_indices, protein_indices, evidence=None):
        compound_indices = np.asarray(compound_indices, dtype=np.int64)
        protein_indices = np.asarray(protein_indices, dtype=np.int64)
        if self.prototype_enabled:
            prototype = support_calibrated_herb_prototype_scores(
                evidence[0] if evidence is not None else self.herb_prototypes,
                self.compound_herb_indices,
                self.compound_herb_mask,
                compound_indices,
                protein_indices,
                prior_strength=self.prototype_prior,
                exclude_query_prior=self.exclude_query_prior,
                raw_scores=self.raw_transfer_scores,
            )
        else:
            prototype = np.zeros(compound_indices.size, dtype=np.float32)
        if self.jaccard_enabled:
            jaccard = loco_jaccard_pair_residuals(
                evidence[1] if evidence is not None else self.jaccard_transfer,
                compound_indices,
                protein_indices,
                exclude_query_prior=self.exclude_query_prior,
                raw_scores=self.raw_transfer_scores,
            )
        else:
            jaccard = np.zeros(compound_indices.size, dtype=np.float32)
        return prototype, jaccard

    def prepareTrainingEpoch(self, epoch):
        self.training_epoch_prepared = True
        self.training_query_mask[:] = False
        # Release the previous dense prototype table before building another block.
        self.training_transfer_state = None
        if not self.training_mask_id:
            return
        queries = self.training_blocks[epoch % len(self.training_blocks)]
        self.training_query_mask[queries] = True
        if self.training_mask_evidence:
            self.training_transfer_state = masked_transfer_state(
                self.hc_matrix, self.cp_matrix, queries
            )

    def _feed(self, compound_indices, protein_indices, labels=None, training=False):
        prototype, jaccard = self.transferResiduals(
            compound_indices, protein_indices
        )
        selected = None
        if training and self.training_mask_id:
            if not self.training_epoch_prepared:
                raise ValueError('Prepare the training epoch before masked batches.')
            selected = self.training_query_mask[np.asarray(compound_indices)]
            if self.training_mask_evidence and np.any(selected):
                if self.training_transfer_state is None:
                    raise ValueError('Prepare the training epoch before masked batches.')
                masked_p, masked_j = self.transferResiduals(
                    np.asarray(compound_indices)[selected],
                    np.asarray(protein_indices)[selected],
                    evidence=self.training_transfer_state,
                )
                prototype[selected], jaccard[selected] = masked_p, masked_j
        feed = {
            self.u_idx: compound_indices,
            self.v_idx: protein_indices,
            self.prototype_residual: prototype,
            self.jaccard_residual: jaccard,
        }
        if labels is not None:
            feed[self.r] = labels
        if selected is not None:
            feed[self.training_id_keep] = (~selected).astype(np.float32)
        if training and self.hc_reconstruction_weight > 0:
            feed.update(self.hcReconstructionFeed(
                *self.hc_reconstruction_sampler.sample(self.hc_reconstruction_samples)))
        return feed

    def fetchModelState(self):
        values = self.sess.run({
            'compound_id': self.final_compound_id,
            'compound_context': self.final_compound_context,
            'protein': self.final_protein,
            'context_interaction': self.context_interaction,
            'protein_context_scale': self.protein_context_scale,
            'prototype_scale': self.prototype_scale,
            'jaccard_scale': self.jaccard_scale,
        })
        return values

    def scorePairs(self, state, compound_indices, protein_indices):
        compound_indices = np.asarray(compound_indices, dtype=np.int64)
        protein_indices = np.asarray(protein_indices, dtype=np.int64)
        protein = state['protein'][protein_indices]
        id_gate = (
            (
                self.compound_support_degrees[compound_indices] > 0
            ).astype(np.float32)
            if self.support_id_gate_enabled else
            np.ones(compound_indices.shape[0], dtype=np.float32)
        )
        context_gate = (
            self.compound_context_available[compound_indices]
            if self.context_expert_enabled else
            np.zeros(compound_indices.shape[0], dtype=np.float32)
        )
        scores = id_gate * np.sum(
            state['compound_id'][compound_indices] * protein, axis=1
        )
        scores += context_gate * np.sum(
            state['compound_context'][compound_indices]
            * protein
            * state['context_interaction'],
            axis=1,
        )
        prototype, jaccard = self.transferResiduals(
            compound_indices, protein_indices
        )
        scores += float(state['prototype_scale'][0]) * prototype
        scores += float(state['jaccard_scale'][0]) * jaccard
        return scores.astype(np.float32)

    def evaluateValidation(self, state, metric):
        if not self.validationData:
            raise ValueError(
                'Early stopping is enabled but no inner validation records '
                'were provided.'
            )
        compounds = np.asarray([
            self.data.compound[str(row[0])] for row in self.validationData
        ], dtype=np.int64)
        proteins = np.asarray([
            self.data.protein[str(row[1])] for row in self.validationData
        ], dtype=np.int64)
        labels = np.asarray([
            1 if float(row[2]) > 0 else 0 for row in self.validationData
        ], dtype=np.int32)
        if np.unique(labels).size < 2:
            raise ValueError(
                'Inner validation must contain both positive and negative '
                'records.'
            )
        logits = self.scorePairs(state, compounds, proteins)
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
        if metric == 'aupr':
            return float(average_precision_score(labels, scores))
        if metric == 'auc':
            return float(roc_auc_score(labels, scores))
        raise ValueError('Unsupported validation metric: %s' % metric)

    def buildTrainingMetadata(self):
        return {
            'model': 'SACTI',
            'model_role': 'independent support-aware context-transfer model',
            'inherits_hdcti': False,
            'encoder': {
                'type': 'sparse_membership_residual_mlp',
                'context_layers': self.context_layers,
                'uses_pagerank': False,
                'uses_self_gating': False,
                'uses_dense_full_attention': False,
                'uses_node_dimension_attention': False,
            },
            'support': {
                'id_gate_enabled': self.support_id_gate_enabled,
                'id_expert': (
                    'active_only_with_inner_train_cp_support'
                    if self.support_id_gate_enabled else 'ungated'
                ),
                'context_expert_enabled': self.context_expert_enabled,
                'context_expert': (
                    'active_when_hc_context_is_observed'
                    if self.context_expert_enabled else 'disabled'
                ),
                'training_positive_edges': int(self.cp_matrix.nnz),
                'supported_compounds': int(np.sum(
                    self.compound_support_degrees > 0
                )),
            },
            'transfer': {
                'prototype': self.prototype_enabled,
                'jaccard': self.jaccard_enabled,
                'prior_strength': self.prototype_prior,
                'exclude_query_prior': self.exclude_query_prior,
                'raw_scores': self.raw_transfer_scores,
                'no_donor_fallback': 'query_excluded_prior' if self.raw_transfer_scores else 'zero',
                'construction': 'fold_local_leave_one_compound_out',
            },
            'hc_reconstruction': {
                'weight': self.hc_reconstruction_weight,
                'samples_per_batch': self.hc_reconstruction_samples,
                'scope': 'inner_train_cp_supported_compounds_only',
                'balanced_pairs_share_masked_context': True,
                'sampler': (dict(self.hc_reconstruction_sampler.metadata)
                            if self.hc_reconstruction_weight > 0 else None),
            },
            'pseudo_cold_training': {
                'mask_id': self.training_mask_id,
                'mask_evidence': self.training_mask_evidence,
                'block_count': self.training_block_count,
                'block_seed': self.training_block_seed,
                'block_indices': [block.tolist() for block in self.training_blocks],
                'masked_records': self.training_masked_records,
                'schedule': 'one_block_per_epoch_cyclic_existing_batches',
                'inference': 'original_inner_train_evidence_and_support_gate',
            },
        }

    def trainModel(self):
        if self.hc_reconstruction_weight > 0 and not len(self.hc_reconstruction_sampler.eligible):
            raise ValueError(
                'No eligible training compounds for H-C reconstruction; inspect side coverage.')
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
                early_stopping['patience'], early_stopping['min_delta']
            )

        current_time = strftime('%Y-%m-%d %H-%M-%S', localtime(time()))
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
            self.prepareTrainingEpoch(epoch)
            for batch_number, batch in enumerate(
                    self.next_batch_pairwise(), start=1):
                batch_start = time()
                compounds, proteins, labels = batch
                feed = self._feed(compounds, proteins, labels, training=True)
                _, loss_value = self.sess.run(
                    [self.train_operation, self.loss], feed_dict=feed
                )
                if self.training_mask_id:
                    self.training_masked_records += int(np.sum(
                        self.training_query_mask[np.asarray(compounds)]
                    ))
                if not np.isfinite(loss_value):
                    raise ValueError(
                        'Training loss became non-finite at epoch %d batch '
                        '%d: %s' % (epoch + 1, batch_number, loss_value)
                    )
                elapsed = time() - train_start
                step = min(
                    epoch * total_batches + batch_number, total_steps
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
                value = self.evaluateValidation(
                    self.fetchModelState(), early_stopping['metric']
                )
                improved, should_stop = tracker.update(value, epoch + 1)
                if improved:
                    saver.save(self.sess, model_path)
                print(
                    'validation: epoch %d %s=%.6f best=%.6f '
                    'best_epoch=%d stale=%d/%d%s' % (
                        epoch + 1,
                        early_stopping['metric'].upper(),
                        value,
                        tracker.best_value,
                        tracker.best_epoch,
                        tracker.stale_checks,
                        tracker.patience,
                        ' improved' if improved else '',
                    )
                )
                if should_stop:
                    print('Early stopping triggered at epoch %d.' % (epoch + 1))
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

        self.state = self.fetchModelState()
        self.u = self.state['compound_id']
        self.i = self.state['protein']
        metadata = self.buildTrainingMetadata()
        metadata['early_stopping'] = self.early_stopping_summary
        metadata['learned_scales'] = {
            'protein_context': float(
                self.state['protein_context_scale'][0]
            ),
            'prototype': float(self.state['prototype_scale'][0]),
            'jaccard': float(self.state['jaccard_scale'][0]),
            'context_weight_mean_abs': float(np.mean(np.abs(
                self.state['context_interaction']
            ))),
        }
        metadata_path = os.path.join(model_dir, self.metadata_filename)
        with open(metadata_path, 'w', encoding='utf-8') as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
        print('SACTI metadata: %s' % metadata_path)
        print('模型权重保存成功: %s' % model_path)

    def predictForPairs(self, compound_indices, protein_indices):
        compound_indices = np.asarray(compound_indices, dtype=np.int64)
        protein_indices = np.asarray(protein_indices, dtype=np.int64)
        parts = []
        for start in range(0, compound_indices.size, self.prediction_batch_size):
            stop = start + self.prediction_batch_size
            parts.append(self.scorePairs(
                self.state,
                compound_indices[start:stop],
                protein_indices[start:stop],
            ))
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts)

    def predictForRanking(self):
        scores = np.empty(
            (self.num_compounds, self.num_proteins), dtype=np.float32
        )
        proteins = np.arange(self.num_proteins, dtype=np.int64)
        for compound in range(self.num_compounds):
            compounds = np.full(
                self.num_proteins, compound, dtype=np.int64
            )
            scores[compound] = self.scorePairs(
                self.state, compounds, proteins
            )
        return scores
