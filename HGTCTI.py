# coding:utf8
import math

import numpy as np

from util.gpu import configure_cuda_environment

configure_cuda_environment()

import tensorflow.compat.v1 as tf

tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()

from base.herbRecommender import herbRecommender
from LightGCNCTI import LightGCNCTI
from util.hgt import build_hgt_relations
from util.rgcn import RELATION_SPECS


class HGTCTI(LightGCNCTI):
    """Sparse same-input heterogeneous graph transformer baseline."""

    checkpoint_filename = 'hgt_cti_model.ckpt'
    metadata_filename = 'hgt_cti.json'
    metadata_label = 'HGT'

    def readConfiguration(self):
        herbRecommender.readConfiguration(self)
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
                'HGT-CTI is a frozen heterogeneous attention baseline; '
                'incompatible settings: %s.' % ', '.join(incompatible)
            )
        self.n_layers = (
            int(self.config['hgt.layers'])
            if self.config.contains('hgt.layers') else 2
        )
        self.n_heads = (
            int(self.config['hgt.heads'])
            if self.config.contains('hgt.heads') else 2
        )
        self.max_neighbors = (
            int(self.config['hgt.max.neighbors'])
            if self.config.contains('hgt.max.neighbors') else 64
        )
        self.sampling_seed = (
            int(self.config['hgt.sampling.seed'])
            if self.config.contains('hgt.sampling.seed') else 2026
        )
        self.activation = (
            str(self.config['hgt.activation']).strip().lower()
            if self.config.contains('hgt.activation') else 'gelu'
        )
        self.objective = (
            str(self.config['hgt.objective']).strip().lower()
            if self.config.contains('hgt.objective') else 'bce'
        )
        self.weight_decay = (
            float(self.config['weight.reg'])
            if self.config.contains('weight.reg') else 0.01
        )

        if self.n_layers < 1:
            raise ValueError('hgt.layers must be at least 1.')
        if self.n_heads < 1:
            raise ValueError('hgt.heads must be at least 1.')
        if self.emb_size % self.n_heads != 0:
            raise ValueError(
                'num.factors must be divisible by hgt.heads.'
            )
        if self.max_neighbors < 0:
            raise ValueError('hgt.max.neighbors must be non-negative.')
        if self.activation != 'gelu':
            raise ValueError(
                'The frozen HGT-CTI baseline requires hgt.activation=gelu.'
            )
        if self.objective != 'bce':
            raise ValueError(
                'The frozen same-input HGT-CTI baseline requires '
                'hgt.objective=bce.'
            )
        if self.weight_decay < 0:
            raise ValueError('weight.reg must be non-negative.')

    @staticmethod
    def _gelu(values):
        coefficient = math.sqrt(2.0 / math.pi)
        return 0.5 * values * (
            1.0 + tf.tanh(
                coefficient * (
                    values + 0.044715 * tf.pow(values, 3)
                )
            )
        )

    @staticmethod
    def _segment_softmax(logits, segment_ids, segment_count, name):
        with tf.name_scope(name):
            maxima = tf.unsorted_segment_max(
                logits,
                segment_ids,
                segment_count,
            )
            stabilized = logits - tf.gather(maxima, segment_ids)
            exponentials = tf.exp(stabilized)
            denominators = tf.unsorted_segment_sum(
                exponentials,
                segment_ids,
                segment_count,
            )
            return exponentials / tf.gather(
                denominators + 1e-12,
                segment_ids,
            )

    def _variable(self, name, shape, initializer=None):
        variable = tf.get_variable(
            name,
            shape=shape,
            initializer=(
                initializer
                if initializer is not None
                else tf.glorot_uniform_initializer()
            ),
        )
        self.hgt_weights[name] = variable
        return variable

    def initModel(self):
        if getattr(self.data, 'protocol', 'legacy') != 'strict':
            raise ValueError('HGT-CTI requires experiment.protocol=strict.')
        if (
            self.config.contains('pair.decoder')
            and str(self.config['pair.decoder']).strip().lower() != 'dot'
        ):
            raise ValueError('HGT-CTI supports only pair.decoder=dot.')

        herbRecommender.initModel(self)
        self.herb_embeddings = tf.Variable(
            tf.truncated_normal(
                shape=[self.num_herbs, self.emb_size],
                stddev=0.05,
            ),
            name='hgt_herb_embeddings',
        )
        self.disease_embeddings = tf.Variable(
            tf.truncated_normal(
                shape=[self.num_diseases, self.emb_size],
                stddev=0.05,
            ),
            name='hgt_disease_embeddings',
        )

        herb_compound_edges = [
            (
                self.data.herb[str(herb_id)],
                self.data.compound[str(compound_id)],
            )
            for herb_id, compound_id, *_ in self.data.hcassociation
        ]
        compound_protein_edges = [
            (
                self.data.compound[str(compound_id)],
                self.data.protein[str(protein_id)],
            )
            for compound_id, protein_id, label in self.data.cpassociation
            if float(label) > 0
        ]
        protein_disease_edges = [
            (
                self.data.protein[str(protein_id)],
                self.data.disease[str(disease_id)],
            )
            for protein_id, disease_id, *_ in self.data.pdassociation
        ]
        self.graph_metadata = build_hgt_relations(
            self.num_herbs,
            self.num_compounds,
            self.num_proteins,
            self.num_diseases,
            herb_compound_edges,
            compound_protein_edges,
            protein_disease_edges,
            max_neighbors=self.max_neighbors,
            seed=self.sampling_seed,
        )

        states = {
            'herb': self.herb_embeddings,
            'compound': self.compound_embeddings,
            'protein': self.protein_embeddings,
            'disease': self.disease_embeddings,
        }
        node_counts = self.graph_metadata['node_counts']
        head_size = self.emb_size // self.n_heads
        self.hgt_weights = {}

        for layer in range(1, self.n_layers + 1):
            projections = {}
            for entity_type, state in states.items():
                projections[entity_type] = {}
                for role in ('query', 'key', 'value'):
                    weight = self._variable(
                        'hgt_%s_%s_layer_%d' % (
                            entity_type,
                            role,
                            layer,
                        ),
                        [self.emb_size, self.emb_size],
                    )
                    projected = tf.matmul(
                        state,
                        weight,
                        name='hgt_%s_%s_projection_layer_%d' % (
                            entity_type,
                            role,
                            layer,
                        ),
                    )
                    projections[entity_type][role] = tf.reshape(
                        projected,
                        [-1, self.n_heads, head_size],
                    )

            incoming = {entity_type: [] for entity_type in states}
            for relation_name, source_type, destination_type in RELATION_SPECS:
                relation = self.graph_metadata['relations'][relation_name]
                source_indices = tf.constant(
                    relation['source_indices'],
                    dtype=tf.int32,
                    name='hgt_%s_sources' % relation_name,
                )
                destination_indices = tf.constant(
                    relation['destination_indices'],
                    dtype=tf.int32,
                    name='hgt_%s_destinations' % relation_name,
                )
                source_keys = tf.gather(
                    projections[source_type]['key'],
                    source_indices,
                )
                source_values = tf.gather(
                    projections[source_type]['value'],
                    source_indices,
                )
                destination_queries = tf.gather(
                    projections[destination_type]['query'],
                    destination_indices,
                )
                attention_weight = self._variable(
                    'hgt_%s_attention_layer_%d' % (
                        relation_name,
                        layer,
                    ),
                    [self.n_heads, head_size, head_size],
                )
                message_weight = self._variable(
                    'hgt_%s_message_layer_%d' % (
                        relation_name,
                        layer,
                    ),
                    [self.n_heads, head_size, head_size],
                )
                relation_prior = self._variable(
                    'hgt_%s_prior_layer_%d' % (
                        relation_name,
                        layer,
                    ),
                    [self.n_heads],
                    initializer=tf.ones_initializer(),
                )
                relation_keys = tf.einsum(
                    'ehd,hdf->ehf',
                    source_keys,
                    attention_weight,
                    name='hgt_%s_relation_keys_layer_%d' % (
                        relation_name,
                        layer,
                    ),
                )
                relation_values = tf.einsum(
                    'ehd,hdf->ehf',
                    source_values,
                    message_weight,
                    name='hgt_%s_relation_values_layer_%d' % (
                        relation_name,
                        layer,
                    ),
                )
                logits = (
                    tf.reduce_sum(
                        destination_queries * relation_keys,
                        axis=2,
                    )
                    * relation_prior
                    / math.sqrt(float(head_size))
                )
                incoming[destination_type].append((
                    destination_indices,
                    logits,
                    relation_values,
                ))

            next_states = {}
            for entity_type, edge_groups in incoming.items():
                destination_indices = tf.concat(
                    [group[0] for group in edge_groups],
                    axis=0,
                    name='hgt_%s_destinations_layer_%d' % (
                        entity_type,
                        layer,
                    ),
                )
                logits = tf.concat(
                    [group[1] for group in edge_groups],
                    axis=0,
                    name='hgt_%s_logits_layer_%d' % (
                        entity_type,
                        layer,
                    ),
                )
                messages = tf.concat(
                    [group[2] for group in edge_groups],
                    axis=0,
                    name='hgt_%s_messages_layer_%d' % (
                        entity_type,
                        layer,
                    ),
                )
                attention = self._segment_softmax(
                    logits,
                    destination_indices,
                    node_counts[entity_type],
                    'hgt_%s_segment_softmax_layer_%d' % (
                        entity_type,
                        layer,
                    ),
                )
                aggregated = tf.unsorted_segment_sum(
                    messages * tf.expand_dims(attention, axis=2),
                    destination_indices,
                    node_counts[entity_type],
                    name='hgt_%s_aggregate_layer_%d' % (
                        entity_type,
                        layer,
                    ),
                )
                aggregated = tf.reshape(
                    aggregated,
                    [-1, self.emb_size],
                )
                output_weight = self._variable(
                    'hgt_%s_output_layer_%d' % (
                        entity_type,
                        layer,
                    ),
                    [self.emb_size, self.emb_size],
                )
                skip = self._variable(
                    'hgt_%s_skip_layer_%d' % (
                        entity_type,
                        layer,
                    ),
                    [1],
                    initializer=tf.zeros_initializer(),
                )
                transformed = tf.matmul(
                    aggregated,
                    output_weight,
                    name='hgt_%s_output_projection_layer_%d' % (
                        entity_type,
                        layer,
                    ),
                )
                gate = tf.sigmoid(skip)
                combined = (
                    gate * transformed
                    + (1.0 - gate) * states[entity_type]
                )
                next_states[entity_type] = self._gelu(combined)
            states = next_states

        self.final_uembedding = states['compound']
        self.final_iembedding = states['protein']
        self.u_embedding = tf.nn.embedding_lookup(
            self.final_uembedding,
            self.u_idx,
        )
        self.v_embedding = tf.nn.embedding_lookup(
            self.final_iembedding,
            self.v_idx,
        )
        self.logits = tf.reduce_sum(
            self.u_embedding * self.v_embedding,
            axis=1,
            name='hgt_pair_logits',
        )
        self.bce_loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.r,
                logits=self.logits,
            )
        )
        embedding_regularization = (
            self.regU * (
                tf.nn.l2_loss(self.compound_embeddings)
                + tf.nn.l2_loss(self.herb_embeddings)
            )
            + self.regI * (
                tf.nn.l2_loss(self.protein_embeddings)
                + tf.nn.l2_loss(self.disease_embeddings)
            )
        )
        weight_regularization = tf.add_n([
            self.weight_decay * tf.nn.l2_loss(weight)
            for weight in self.hgt_weights.values()
        ])
        self.regularization_loss = (
            embedding_regularization + weight_regularization
        )
        self.loss = self.bce_loss + self.regularization_loss
        self.train_operation = tf.train.AdamOptimizer(
            self.lRate
        ).minimize(self.loss)

        print(
            'HGT-CTI: layers=%d heads=%d activation=%s relations=6 '
            'objective=%s normalization=%s.' % (
                self.n_layers,
                self.n_heads,
                self.activation,
                self.objective,
                self.graph_metadata['attention_normalization'],
            )
        )
        print(
            'HGT training graph: directed_edges=%d/%d max_neighbors=%d '
            'seed=%d; C-P source=inner_train_positive.' % (
                self.graph_metadata['sampled_directed_edges'],
                self.graph_metadata['original_directed_edges'],
                self.max_neighbors,
                self.sampling_seed,
            )
        )

    def buildTrainingMetadata(self):
        relations = {}
        for relation_name, relation in (
                self.graph_metadata['relations'].items()):
            relations[relation_name] = {
                key: value
                for key, value in relation.items()
                if key not in {
                    'source_indices',
                    'destination_indices',
                }
            }
            relations[relation_name]['shape'] = list(
                relations[relation_name]['shape']
            )
        return {
            'model_role': (
                'HGT-CTI same-input deterministic-neighbor BCE adaptation'
            ),
            'layers': self.n_layers,
            'heads': self.n_heads,
            'activation': self.activation,
            'relations': len(RELATION_SPECS),
            'objective': self.objective,
            'decoder': 'dot',
            'attention_normalization': self.graph_metadata[
                'attention_normalization'
            ],
            'sampling': self.graph_metadata['sampling'],
            'graph_source': {
                'H_C': 'fixed_side_information',
                'C_P': 'strict_inner_train_positive_C-P',
                'P_D': 'fixed_side_information',
            },
            'node_counts': self.graph_metadata['node_counts'],
            'source_edge_counts': self.graph_metadata[
                'source_edge_counts'
            ],
            'sampled_source_edge_counts': self.graph_metadata[
                'sampled_source_edge_counts'
            ],
            'original_directed_edges': self.graph_metadata[
                'original_directed_edges'
            ],
            'sampled_directed_edges': self.graph_metadata[
                'sampled_directed_edges'
            ],
            'relation_graphs': relations,
            'weight_regularization': self.weight_decay,
        }
