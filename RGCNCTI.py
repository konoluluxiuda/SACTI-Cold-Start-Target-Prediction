# coding:utf8
import numpy as np

from util.gpu import configure_cuda_environment

configure_cuda_environment()

import tensorflow.compat.v1 as tf

tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()

from base.herbRecommender import herbRecommender
from LightGCNCTI import LightGCNCTI
from util.rgcn import RELATION_SPECS, build_rgcn_relations


class RGCNCTI(LightGCNCTI):
    """Same-input relation-aware heterogeneous graph baseline."""

    checkpoint_filename = 'rgcn_cti_model.ckpt'
    metadata_filename = 'rgcn_cti.json'
    metadata_label = 'R-GCN'

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
                'R-GCN-CTI is a frozen heterogeneous graph baseline; '
                'incompatible settings: %s.' % ', '.join(incompatible)
            )
        self.n_layers = (
            int(self.config['rgcn.layers'])
            if self.config.contains('rgcn.layers') else 2
        )
        if self.n_layers < 1:
            raise ValueError('rgcn.layers must be at least 1.')
        self.objective = (
            str(self.config['rgcn.objective']).strip().lower()
            if self.config.contains('rgcn.objective') else 'bce'
        )
        if self.objective != 'bce':
            raise ValueError(
                'The frozen same-input R-GCN-CTI baseline requires '
                'rgcn.objective=bce.'
            )
        self.activation = (
            str(self.config['rgcn.activation']).strip().lower()
            if self.config.contains('rgcn.activation') else 'relu'
        )
        if self.activation != 'relu':
            raise ValueError(
                'The frozen R-GCN-CTI baseline requires '
                'rgcn.activation=relu.'
            )
        self.weight_decay = (
            float(self.config['weight.reg'])
            if self.config.contains('weight.reg') else 0.01
        )
        if self.weight_decay < 0:
            raise ValueError('weight.reg must be non-negative.')

    @staticmethod
    def _sparse_tensor(relation):
        return tf.sparse_reorder(tf.SparseTensor(
            relation['indices'],
            relation['values'],
            relation['shape'],
        ))

    def initModel(self):
        if getattr(self.data, 'protocol', 'legacy') != 'strict':
            raise ValueError('R-GCN-CTI requires experiment.protocol=strict.')
        if (
            self.config.contains('pair.decoder')
            and str(self.config['pair.decoder']).strip().lower() != 'dot'
        ):
            raise ValueError('R-GCN-CTI supports only pair.decoder=dot.')

        herbRecommender.initModel(self)
        self.herb_embeddings = tf.Variable(
            tf.truncated_normal(
                shape=[self.num_herbs, self.emb_size], stddev=0.05
            ),
            name='rgcn_herb_embeddings',
        )
        self.disease_embeddings = tf.Variable(
            tf.truncated_normal(
                shape=[self.num_diseases, self.emb_size], stddev=0.05
            ),
            name='rgcn_disease_embeddings',
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
        self.graph_metadata = build_rgcn_relations(
            self.num_herbs,
            self.num_compounds,
            self.num_proteins,
            self.num_diseases,
            herb_compound_edges,
            compound_protein_edges,
            protein_disease_edges,
        )
        relation_tensors = {
            name: self._sparse_tensor(relation)
            for name, relation in self.graph_metadata['relations'].items()
        }

        states = {
            'herb': self.herb_embeddings,
            'compound': self.compound_embeddings,
            'protein': self.protein_embeddings,
            'disease': self.disease_embeddings,
        }
        self.rgcn_weights = {}
        for layer in range(1, self.n_layers + 1):
            self_weight = tf.get_variable(
                'rgcn_self_layer_%d' % layer,
                shape=[self.emb_size, self.emb_size],
                initializer=tf.glorot_uniform_initializer(),
            )
            self.rgcn_weights['self_layer_%d' % layer] = self_weight
            incoming = {
                entity_type: [
                    tf.matmul(
                        embedding,
                        self_weight,
                        name='rgcn_self_%s_layer_%d' % (
                            entity_type, layer
                        ),
                    )
                ]
                for entity_type, embedding in states.items()
            }
            for relation_name, source_type, destination_type in RELATION_SPECS:
                relation_weight = tf.get_variable(
                    'rgcn_%s_layer_%d' % (relation_name, layer),
                    shape=[self.emb_size, self.emb_size],
                    initializer=tf.glorot_uniform_initializer(),
                )
                self.rgcn_weights[
                    '%s_layer_%d' % (relation_name, layer)
                ] = relation_weight
                transformed_source = tf.matmul(
                    states[source_type],
                    relation_weight,
                    name='rgcn_transform_%s_layer_%d' % (
                        relation_name, layer
                    ),
                )
                incoming[destination_type].append(
                    tf.sparse_tensor_dense_matmul(
                        relation_tensors[relation_name],
                        transformed_source,
                        name='rgcn_message_%s_layer_%d' % (
                            relation_name, layer
                        ),
                    )
                )

            next_states = {}
            for entity_type, messages in incoming.items():
                aggregate = tf.add_n(
                    messages,
                    name='rgcn_aggregate_%s_layer_%d' % (
                        entity_type, layer
                    ),
                )
                if layer < self.n_layers:
                    aggregate = tf.nn.relu(
                        aggregate,
                        name='rgcn_relu_%s_layer_%d' % (
                            entity_type, layer
                        ),
                    )
                next_states[entity_type] = aggregate
            states = next_states

        self.final_uembedding = states['compound']
        self.final_iembedding = states['protein']
        self.u_embedding = tf.nn.embedding_lookup(
            self.final_uembedding, self.u_idx
        )
        self.v_embedding = tf.nn.embedding_lookup(
            self.final_iembedding, self.v_idx
        )
        self.logits = tf.reduce_sum(
            self.u_embedding * self.v_embedding,
            axis=1,
            name='rgcn_pair_logits',
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
            for weight in self.rgcn_weights.values()
        ])
        self.regularization_loss = (
            embedding_regularization + weight_regularization
        )
        self.loss = self.bce_loss + self.regularization_loss
        self.train_operation = tf.train.AdamOptimizer(
            self.lRate
        ).minimize(self.loss)

        counts = self.graph_metadata['source_edge_counts']
        print(
            'R-GCN-CTI: layers=%d relations=6 activation=%s '
            'objective=%s normalization=%s.' % (
                self.n_layers,
                self.activation,
                self.objective,
                self.graph_metadata['normalization'],
            )
        )
        print(
            'R-GCN training graph: H-C=%d C-P=%d P-D=%d; '
            'C-P source=inner_train_positive.' % (
                counts['H_C'], counts['C_P'], counts['P_D']
            )
        )

    def buildTrainingMetadata(self):
        relations = {}
        for relation_name, relation in (
                self.graph_metadata['relations'].items()):
            relations[relation_name] = {
                key: value
                for key, value in relation.items()
                if key not in {'indices', 'values'}
            }
            relations[relation_name]['shape'] = list(
                relations[relation_name]['shape']
            )
        return {
            'model_role': 'R-GCN-CTI same-input BCE adaptation',
            'layers': self.n_layers,
            'relations': len(RELATION_SPECS),
            'activation': self.activation,
            'objective': self.objective,
            'decoder': 'dot',
            'normalization': self.graph_metadata['normalization'],
            'graph_source': {
                'H_C': 'fixed_side_information',
                'C_P': 'strict_inner_train_positive_C-P',
                'P_D': 'fixed_side_information',
            },
            'node_counts': self.graph_metadata['node_counts'],
            'source_edge_counts': self.graph_metadata[
                'source_edge_counts'
            ],
            'relation_graphs': relations,
            'weight_regularization': self.weight_decay,
        }
