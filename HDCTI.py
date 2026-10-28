# coding:utf8
# Derived from https://github.com/tong87-bio/HDCTI (MIT License).
# This repository adds strict fold-local evaluation, Hctx-P, SDIS, and SCHPT.
import json
import os
from util.gpu import configure_cuda_environment
configure_cuda_environment()

import tensorflow.compat.v1 as tf #使用1.0版本的方法
tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior() #禁用2.0版本的方法
from base.herbRecommender import herbRecommender
from scipy.sparse import coo_matrix,hstack
#import tensorflow as tf
import numpy as np
from math import sqrt
import pandas as pds
import scipy.sparse as sp
# from spektral.layers import GraphAttention
from time import strftime,localtime,time
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import Dropout

import networkx as nx
from util.graph import bipartite_pagerank
from util.model_components import (
    EarlyStoppingTracker,
    build_regularization_loss,
    context_interaction_pair_scores,
    resolve_context_terms,
    resolve_context_mask_training,
    resolve_counterfactual_context,
    resolve_early_stopping,
    resolve_encoder_profile,
    resolve_global_token_attention,
    resolve_hplga,
    resolve_hyperedge_attention,
    resolve_herb_context_attention,
    resolve_herb_prototype_transfer,
    resolve_hc_ranking_distillation,
    resolve_safe_listwise_reranker,
    resolve_inductive_context,
    resolve_pair_decoder,
    resolve_support_experts,
    resolve_support_router,
    resolve_support_state_routing,
    support_conditioned_cold_gate,
    support_decoupled_base_gate,
    support_state_pair_gates,
    target_conditioned_herb_contexts,
)
from util.support_router import (
    inverse_softplus,
    monotonic_context_gate,
    softplus,
)
from util.hyperedge_attention import (
    hyperedge_specificity_prior,
    ordered_incidence_ids,
)
from util.global_token_attention import (
    global_token_attention_complexity,
    summarize_global_token_attention,
)
from util.hplga import (
    hplga_complexity,
    hypergraph_pagerank,
    pagerank_linear_attention_tf,
)
from util.herb_prototype_transfer import (
    build_support_calibrated_herb_prototypes,
    support_calibrated_herb_prototype_scores,
)
from util.dual_granularity_transfer import (
    build_loco_jaccard_transfer,
    loco_jaccard_pair_residuals,
)
from util.hc_ranking_distillation import build_distillation_candidate_batch
from util.safe_listwise_reranker import (
    build_safe_listwise_candidate_batch,
    build_support_gate_features,
    full_candidate_base_scores,
)
from util.checkpoint_ranking import evaluate_fixed_candidate_ranking
from sklearn.metrics import average_precision_score, roc_auc_score
# tf.compat.v1.set_random_seed(4321)
from util.io import FileIO

TARGET_HERB_ATTENTION_MODES = {
    'target_attention',
    'target_residual_attention',
}

class HDCTI(herbRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(HDCTI, self).__init__(conf,trainingSet,testSet,fold)


    def buildAdjacencyMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            # symmetric matrix，对称矩阵
            if int(pair[2])!=0:
                row += [self.data.compound[pair[0]]]
                col += [self.data.protein[pair[1]]]
                entries += [1]
        # u_i_adj = coo_matrix((entries, (row, col)), shape=(self.num_herbs,self.num_diseases),dtype=np.float32)
        u_i_adj = coo_matrix((entries,(row,col)), shape=(self.num_compounds, self.num_proteins),dtype=np.float32)
        return u_i_adj

    def buildhcAdjacencyMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.hcassociation:
            # symmetric matrix
            #x=random.randint(0,1008)
            #y=random.randint(0,1192)
            #row += [x]
            #col += [y]
            row += [self.data.herb[pair[0]]]
            col += [self.data.compound[pair[1]]]
            entries += [1]
        u_i_adj = coo_matrix((entries, (row, col)), shape=(self.num_herbs,self.num_compounds),dtype=np.float32)
        return u_i_adj

    def buildcpAdjacencyMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.cpassociation:
            # symmetric matrix
            row += [self.data.compound[pair[0]]]
            col += [self.data.protein[pair[1]]]
            entries += [1]
        u_i_adj = coo_matrix((entries, (row, col)), shape=(self.num_compounds,self.num_proteins),dtype=np.float32)
        return u_i_adj

    def buildpdAdjacencyMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.pdassociation:
            # symmetric matrix
            #x=random.randint(0,7257)
            #y=random.randint(0,11070)
            #row += [x]
            #col += [y]
            row += [self.data.protein[pair[0]]]
            col += [self.data.disease[pair[1]]]
            entries += [1]
        u_i_adj = coo_matrix((entries, (row, col)), shape=(self.num_proteins,self.num_diseases),dtype=np.float32)
        return u_i_adj
    def buildJointAdjacency(self):
        indices = [[self.data.herb[item[0]], self.data.item[item[1]]] for item in self.data.trainingData]
        values = [float(item[2]) / sqrt(len(self.data.trainSet_u[item[0]])) / sqrt(len(self.data.trainSet_i[item[1]]))
                  for item in self.data.trainingData]
        norm_adj = tf.SparseTensor(indices=indices, values=values,
                                   dense_shape=[self.num_herbs, self.num_diseases])
        return norm_adj

    def buildGraphAndPageRank(self, adjacency_matrix):
        G = nx.Graph()
        coo = adjacency_matrix.tocoo()
        for i, j, v in zip(coo.row, coo.col, coo.data):
            G.add_edge(i, j, weight=v)
        pr = nx.pagerank(G, alpha=0.85)
        return pr
    # def buildJointAdjacency(self):    #建立联合邻接矩阵
    #     indices = [[self.data.compound[item[0]], self.data.item[item[1]]] for item in self.data.trainingData]
    #     values = [float(item[2]) / sqrt(len(self.data.trainSet_u[item[0]])) / sqrt(len(self.data.trainSet_i[item[1]]))
    #               for item in self.data.trainingData]
    #     norm_adj = tf.SparseTensor(indices=indices, values=values,
    #                                dense_shape=[self.num_compounds, self.num_proteins])
    #     return norm_adj

    def initModel(self):
        super(HDCTI, self).initModel()

        def safe_reciprocal(values):
            values = np.asarray(values, dtype=np.float32)
            result = np.zeros_like(values)
            np.divide(1.0, values, out=result, where=values != 0)
            return result

        #Build adjacency matrix
        A = self.buildAdjacencyMatrix()
        #A=A.dot(A.transpose().dot(A))
        cp=self.buildcpAdjacencyMatrix()
        pd=self.buildpdAdjacencyMatrix()
        hc=self.buildhcAdjacencyMatrix()



        
        # 计算 hc 的转置
        H_c = hc.transpose()
        # 计算 H_c 中每一行的和，并将其形状调整为 (1, -1)
        D_hc_v = H_c.sum(axis=1).reshape(1, -1)
        # 计算 H_c 中每一列的和，并将其形状调整为 (1, -1)
        D_hc_e = H_c.sum(axis=0).reshape(1, -1)
        # 计算边的归一化矩阵
        temp1 = (H_c.multiply(safe_reciprocal(D_hc_e))).transpose()
        # 计算节点的归一化矩阵
        temp2 = (H_c.transpose().multiply(safe_reciprocal(D_hc_v))).transpose()
        # 将边的矩阵转换为 COO 格式
        edge = temp1.tocoo()
        # 将节点的矩阵转换为 COO 格式
        node = temp2.tocoo()
        # 获取边和节点的索引
        edge_indices = np.asarray([edge.row, edge.col], dtype=np.int64).T
        node_indices = np.asarray([node.row, node.col], dtype=np.int64).T
        # 构建稀疏张量表示边和节点
        H_e = tf.sparse_reorder(
            tf.SparseTensor(edge_indices, edge.data.astype(np.float32), edge.shape)
        )
        H_n = tf.sparse_reorder(
            tf.SparseTensor(node_indices, node.data.astype(np.float32), node.shape)
        )
        compound_herbs = [[] for _ in range(self.num_compounds)]
        hc_coo = hc.tocoo()
        for herb_index, compound_index in zip(hc_coo.row, hc_coo.col):
            compound_herbs[int(compound_index)].append(int(herb_index))
        compound_herbs = [sorted(set(indices)) for indices in compound_herbs]
        self.compound_herbs = compound_herbs
        self.max_compound_herbs = max(
            [len(indices) for indices in compound_herbs] + [1]
        )
        herb_sentinel = self.num_herbs
        self.compound_herb_indices = np.full(
            (self.num_compounds, self.max_compound_herbs),
            herb_sentinel,
            dtype=np.int32,
        )
        self.compound_herb_mask = np.zeros(
            (self.num_compounds, self.max_compound_herbs), dtype=np.float32
        )
        for compound_index, herb_indices in enumerate(compound_herbs):
            degree = len(herb_indices)
            if degree:
                self.compound_herb_indices[compound_index, :degree] = herb_indices
                self.compound_herb_mask[compound_index, :degree] = 1.0
        hc_attention_edge_ids = np.asarray(hc_coo.row, dtype=np.int32)
        hc_attention_node_ids = np.asarray(hc_coo.col, dtype=np.int32)
        hc_attention_ids = ordered_incidence_ids(
            hc_attention_edge_ids, hc_attention_node_ids
        )
        hc_edge_degrees = np.bincount(
            hc_attention_edge_ids, minlength=self.num_herbs
        ).astype(np.float32)
        self.compound_inductive_context_available = np.asarray([
            any(hc_edge_degrees[herb_index] > 1 for herb_index in herb_indices)
            for herb_indices in self.compound_herbs
        ], dtype=np.float32)
        hc_specificity_prior = hyperedge_specificity_prior(
            hc_edge_degrees, self.num_compounds
        )
        
        # 获取 pd 的矩阵
        P_d = pd
        # 计算 P_d 中每一行的和，并将其形状调整为 (1, -1)
        D_P_v = P_d.sum(axis=1).reshape(1, -1)
        # 计算 P_d 中每一列的和，并将其形状调整为 (1, -1)
        D_P_e = P_d.sum(axis=0).reshape(1, -1)
        # 计算边的矩阵
        temp1 = (P_d.multiply(safe_reciprocal(D_P_e))).transpose()
        # 计算节点的矩阵
        temp2 = (P_d.transpose().multiply(safe_reciprocal(D_P_v))).transpose()
        # 将边的矩阵转换为 COO 格式
        pd_edge = temp1.tocoo()
        # 将节点的矩阵转换为 COO 格式
        pd_node = temp2.tocoo()
        # 获取边和节点的索引
        A_pde = np.asarray([pd_edge.row, pd_edge.col], dtype=np.int64).T
        A_pdn = np.asarray([pd_node.row, pd_node.col], dtype=np.int64).T
        # 构建稀疏张量表示边和节点
        P_e = tf.sparse_reorder(
            tf.SparseTensor(A_pde, pd_edge.data.astype(np.float32), pd_edge.shape)
        )
        P_n = tf.sparse_reorder(
            tf.SparseTensor(A_pdn, pd_node.data.astype(np.float32), pd_node.shape)
        )
        pd_coo = pd.tocoo()
        pd_attention_edge_ids = np.asarray(pd_coo.col, dtype=np.int32)
        pd_attention_node_ids = np.asarray(pd_coo.row, dtype=np.int32)
        pd_attention_ids = ordered_incidence_ids(
            pd_attention_edge_ids, pd_attention_node_ids
        )
        pd_edge_degrees = np.bincount(
            pd_attention_edge_ids, minlength=self.num_diseases
        ).astype(np.float32)
        pd_specificity_prior = hyperedge_specificity_prior(
            pd_edge_degrees, self.num_proteins
        )
        self.hyperedge_attention_structure = {
            'hc': {
                'nodes': int(self.num_compounds),
                'hyperedges': int(self.num_herbs),
                'incidences': int(hc_attention_edge_ids.size),
                'specificity_prior_min': float(np.min(hc_specificity_prior)),
                'specificity_prior_max': float(np.max(hc_specificity_prior)),
            },
            'pd': {
                'nodes': int(self.num_proteins),
                'hyperedges': int(self.num_diseases),
                'incidences': int(pd_attention_edge_ids.size),
                'specificity_prior_min': float(np.min(pd_specificity_prior)),
                'specificity_prior_max': float(np.max(pd_specificity_prior)),
            },
        }




        #Build network
        self.isTraining = tf.placeholder(tf.int32)
        self.isTraining = tf.cast(self.isTraining, tf.bool)
        #initializer = tf.contrib.layers.xavier_initializer()
        initializer =tf.keras.initializers.glorot_normal()
        self.n_layer = 2
        self.weights={}
        self.safe_listwise_weights = {}
        self.attention_weights = {}
        self.encoder_profile = resolve_encoder_profile(self.config)
        self.context_terms = resolve_context_terms(self.config)
        self.use_context_interaction = any(self.context_terms.values())
        self.herb_context_attention = resolve_herb_context_attention(self.config)
        self.counterfactual_context = resolve_counterfactual_context(self.config)
        self.context_mask_training = resolve_context_mask_training(self.config)
        self.support_router = resolve_support_router(self.config)
        self.support_experts = resolve_support_experts(self.config)
        self.support_state_routing = resolve_support_state_routing(
            self.config
        )
        self.hyperedge_attention = resolve_hyperedge_attention(self.config)
        self.global_token_attention = resolve_global_token_attention(self.config)
        self.hplga = resolve_hplga(self.config)
        self.inductive_context = resolve_inductive_context(self.config)
        self.pair_decoder = resolve_pair_decoder(self.config)
        self.herb_prototype_transfer = resolve_herb_prototype_transfer(
            self.config
        )
        self.hc_ranking_distillation = resolve_hc_ranking_distillation(
            self.config, self.herb_prototype_transfer
        )
        self.safe_listwise_reranker = resolve_safe_listwise_reranker(
            self.config,
            self.herb_prototype_transfer,
            self.hc_ranking_distillation,
        )
        if self.encoder_profile['external_baseline']:
            incompatible_components = []
            if getattr(self.data, 'protocol', 'legacy') != 'strict':
                incompatible_components.append('experiment.protocol!=strict')
            if self.use_context_interaction:
                incompatible_components.append('context.interaction')
            if self.counterfactual_context['enabled']:
                incompatible_components.append('counterfactual.context')
            if self.context_mask_training['enabled']:
                incompatible_components.append('context.mask.training')
            if self.support_router['enabled']:
                incompatible_components.append('support.router')
            if self.support_experts['enabled']:
                incompatible_components.append('support.experts')
            if self.support_state_routing['enabled']:
                incompatible_components.append('support.state.routing')
            if self.hyperedge_attention['enabled']:
                incompatible_components.append('hyperedge.attention')
            if self.global_token_attention['enabled']:
                incompatible_components.append('global.token.attention')
            if self.hplga['enabled']:
                incompatible_components.append('hplga.enabled')
            if self.inductive_context['enabled']:
                incompatible_components.append('inductive.context')
            if self.herb_context_attention['mode'] != 'static':
                incompatible_components.append('context.herb_protein.mode')
            if self.pair_decoder['type'] != 'dot':
                incompatible_components.append('pair.decoder')
            if self.herb_prototype_transfer['enabled']:
                incompatible_components.append('herb.prototype.transfer')
            if self.safe_listwise_reranker['enabled']:
                incompatible_components.append('safe.listwise.reranker')
            if (
                not self.config.contains('attention.max.nodes')
                or int(self.config['attention.max.nodes']) != 0
            ):
                incompatible_components.append('attention.max.nodes')
            if incompatible_components:
                raise ValueError(
                    'dual_hgnn_cti is a frozen same-input baseline; incompatible '
                    'settings: %s.' % ', '.join(incompatible_components)
                )
        print(
            'Encoder profile: %s (self_gating=%s, pagerank=%s, '
            'dense_full_attention=%s, node_dimension_attention=%s).' % (
                self.encoder_profile['name'],
                'on' if self.encoder_profile['use_self_gating'] else 'off',
                'on' if self.encoder_profile['use_pagerank'] else 'off',
                'on' if self.encoder_profile[
                    'use_dense_full_attention'
                ] else 'off',
                'on' if self.encoder_profile[
                    'use_node_dimension_attention'
                ] else 'off',
            )
        )
        active_attention_encoders = sum([
            int(self.hyperedge_attention['enabled']),
            int(self.global_token_attention['enabled']),
            int(self.hplga['enabled']),
        ])
        if active_attention_encoders > 1:
            raise ValueError(
                'Factorized hyperedge attention, HILGA, and HPLGA are '
                'mutually exclusive encoder variants.'
            )
        if self.support_router['enabled']:
            if getattr(self.data, 'protocol', 'legacy') != 'strict':
                raise ValueError(
                    'Support routing requires experiment.protocol=strict.'
                )
            if self.herb_context_attention['mode'] != 'static':
                raise ValueError('Support routing requires static Hctx-P.')
            if self.context_terms != {
                    'compound_disease': False,
                    'herb_protein': True,
                    'herb_disease': False}:
                raise ValueError(
                    'Support routing requires HerbOnly context terms.'
                )
            if self.context_mask_training['enabled']:
                raise ValueError(
                    'Support routing cannot be combined with context-masked training.'
                )
            if self.data.pseudo_cold_info is None:
                raise ValueError(
                    'Support routing requires deterministic pseudo-cold graph masking.'
                )
        if self.support_experts['enabled']:
            if getattr(self.data, 'protocol', 'legacy') != 'strict':
                raise ValueError(
                    'Support-conditioned experts require experiment.protocol=strict.'
                )
            if self.support_router['enabled']:
                raise ValueError(
                    'Support-conditioned experts cannot use the rejected support router.'
                )
            if self.inductive_context['enabled']:
                raise ValueError(
                    'SCHE replaces standalone inductive.context scoring.'
                )
            if self.herb_context_attention['mode'] != 'static':
                raise ValueError(
                    'Support-conditioned experts require static Hctx-P.'
                )
            if self.context_terms != {
                    'compound_disease': False,
                    'herb_protein': True,
                    'herb_disease': False}:
                raise ValueError(
                    'Support-conditioned experts require HerbOnly context terms.'
                )
            if not self.counterfactual_context['enabled']:
                raise ValueError(
                    'The frozen SCHE Pilot requires counterfactual.context=True.'
                )
            if self.context_mask_training['enabled']:
                raise ValueError(
                    'Support-conditioned experts cannot use CMIT.'
                )
            if self.hyperedge_attention['enabled']:
                raise ValueError(
                    'Support-conditioned experts require standard hypergraph propagation.'
                )
            if self.global_token_attention['enabled']:
                raise ValueError(
                    'Support-conditioned experts cannot use HILGA.'
                )
            if self.pair_decoder['type'] != 'dot':
                raise ValueError(
                    'The frozen SCHE Pilot requires pair.decoder=dot.'
                )
            if self.data.pseudo_cold_info is None:
                raise ValueError(
                    'Support-conditioned experts require deterministic '
                    'pseudo-cold graph masking.'
                )
            if (
                not self.config.contains('attention.max.nodes')
                or int(self.config['attention.max.nodes']) != 0
            ):
                raise ValueError(
                    'Support-conditioned experts require attention.max.nodes=0.'
                )
        if self.support_state_routing['enabled']:
            if getattr(self.data, 'protocol', 'legacy') != 'strict':
                raise ValueError(
                    'Support-state routing requires experiment.protocol=strict.'
                )
            if self.validationAggregation != 'macro_support_states':
                raise ValueError(
                    'Support-state routing requires the shared four-state '
                    'validation protocol.'
                )
            if self.herb_context_attention['mode'] != 'static':
                raise ValueError(
                    'Support-state routing requires static side contexts.'
                )
            if self.context_terms != {
                    'compound_disease': False,
                    'herb_protein': True,
                    'herb_disease': True}:
                raise ValueError(
                    'Support-state routing requires Hctx-P and Hctx-Dctx only.'
                )
            incompatible_components = []
            if self.support_router['enabled']:
                incompatible_components.append('support.router')
            if self.support_experts['enabled']:
                incompatible_components.append('support.experts')
            if self.inductive_context['enabled']:
                incompatible_components.append('inductive.context')
            if self.counterfactual_context['enabled']:
                incompatible_components.append('counterfactual.context')
            if self.context_mask_training['enabled']:
                incompatible_components.append('context.mask.training')
            if self.hyperedge_attention['enabled']:
                incompatible_components.append('hyperedge.attention')
            if self.global_token_attention['enabled']:
                incompatible_components.append('global.token.attention')
            if self.pair_decoder['type'] != 'dot':
                incompatible_components.append('pair.decoder')
            if (
                not self.config.contains('attention.max.nodes')
                or int(self.config['attention.max.nodes']) != 0
            ):
                incompatible_components.append('attention.max.nodes')
            if incompatible_components:
                raise ValueError(
                    'The initial support-state routing Gate requires an '
                    'isolated strict configuration; incompatible settings: '
                    '%s.' % ', '.join(incompatible_components)
                )
        if self.inductive_context['enabled']:
            if getattr(self.data, 'protocol', 'legacy') != 'strict':
                raise ValueError(
                    'Inductive context scoring requires experiment.protocol=strict.'
                )
            if self.herb_context_attention['mode'] != 'static':
                raise ValueError(
                    'Inductive context scoring requires static Hctx-P.'
                )
            if self.context_terms != {
                    'compound_disease': False,
                    'herb_protein': True,
                    'herb_disease': False}:
                raise ValueError(
                    'Inductive context scoring requires HerbOnly context terms.'
                )
            if (
                self.counterfactual_context['enabled']
                and self.inductive_context['self_excluded']
            ):
                raise ValueError(
                    'CHCR cannot be combined with the rejected self-excluded '
                    'H-C readout.'
                )
            if self.context_mask_training['enabled']:
                raise ValueError(
                    'Inductive context scoring cannot be combined with CMIT.'
                )
            if self.support_router['enabled']:
                raise ValueError(
                    'Inductive context scoring cannot use the support router.'
                )
            if self.hyperedge_attention['enabled']:
                raise ValueError(
                    'Self-excluded H-C readout requires standard hypergraph propagation.'
                )
            if self.global_token_attention['enabled']:
                raise ValueError(
                    'Inductive context scoring cannot use HILGA.'
                )
            if (
                not self.config.contains('attention.max.nodes')
                or int(self.config['attention.max.nodes']) != 0
            ):
                raise ValueError(
                    'Inductive context scoring requires attention.max.nodes=0.'
                )
        if self.herb_prototype_transfer['enabled']:
            incompatible_components = []
            if getattr(self.data, 'protocol', 'legacy') != 'strict':
                incompatible_components.append('experiment.protocol')
            if (
                self.encoder_profile['use_pagerank']
                != self.herb_prototype_transfer['replace_compound_pagerank']
            ):
                incompatible_components.append(
                    'herb.prototype.replace.compound.pagerank'
                )
            if self.context_terms != {
                    'compound_disease': False,
                    'herb_protein': True,
                    'herb_disease': False}:
                incompatible_components.append('context.interaction')
            if self.herb_context_attention['mode'] != 'static':
                incompatible_components.append('context.herb_protein.mode')
            if (
                not self.inductive_context['enabled']
                or not self.inductive_context['suppress_base_zero_support']
                or self.inductive_context['self_excluded']
            ):
                incompatible_components.append('inductive.context')
            if self.counterfactual_context['enabled']:
                incompatible_components.append('counterfactual.context')
            if self.context_mask_training['enabled']:
                incompatible_components.append('context.mask.training')
            if self.support_router['enabled'] or self.support_experts['enabled']:
                incompatible_components.append('support.router/experts')
            if self.support_state_routing['enabled']:
                incompatible_components.append('support.state.routing')
            if (
                self.hyperedge_attention['enabled']
                or self.global_token_attention['enabled']
                or self.hplga['enabled']
            ):
                incompatible_components.append('alternative.encoder')
            if self.pair_decoder['type'] != 'dot':
                incompatible_components.append('pair.decoder')
            if (
                not self.config.contains('attention.max.nodes')
                or int(self.config['attention.max.nodes']) != 0
            ):
                incompatible_components.append('attention.max.nodes')
            if incompatible_components:
                raise ValueError(
                    'The frozen herb prototype Pilot requires Hctx-P + SDIS '
                    'under the strict no-dense-attention protocol; incompatible '
                    'settings: %s.' % ', '.join(incompatible_components)
                )
            if self.encoder_profile['use_pagerank']:
                print(
                    'Herb prototype transfer: mode=%s prior=%g; replacing '
                    'compound PageRank while retaining protein P-D PageRank.' % (
                        self.herb_prototype_transfer['mode'],
                        self.herb_prototype_transfer['prior_strength'],
                    )
                )
            else:
                print(
                    'Herb prototype transfer: mode=%s prior=%g; independent '
                    'of PageRank under the portability-audit encoder.' % (
                        self.herb_prototype_transfer['mode'],
                        self.herb_prototype_transfer['prior_strength'],
                    )
                )
            if self.herb_prototype_transfer['dual_granularity']:
                if (
                    self.herb_prototype_transfer['dual_mode']
                    == 'independent_scales'
                ):
                    print(
                        'Dual-granularity transfer: independent zero-initialized '
                        'SCHPT and H-C-Jaccard scales; fold-local LOCO propagation.'
                    )
                else:
                    print(
                        'Dual-granularity transfer: SCHPT=%g H-C-Jaccard=%g; '
                        'fold-local LOCO propagation.' % (
                            1.0 - self.herb_prototype_transfer['jaccard_blend'],
                            self.herb_prototype_transfer['jaccard_blend'],
                        )
                    )
        if self.hyperedge_attention['enabled']:
            print(
                'Hyperedge attention: mode=%s H-C=%s P-D=%s '
                'temperature=%g prior_scale=%g.' % (
                    self.hyperedge_attention['mode'],
                    'on' if self.hyperedge_attention['hc_enabled'] else 'off',
                    'on' if self.hyperedge_attention['pd_enabled'] else 'off',
                    self.hyperedge_attention['temperature'],
                    self.hyperedge_attention['prior_scale'],
                )
            )
        if self.global_token_attention['enabled']:
            if self.emb_size % self.global_token_attention['heads'] != 0:
                raise ValueError(
                    'num.factors must be divisible by '
                    'global.token.attention.heads.'
                )
            print(
                'HILGA global token attention: H-C=%s P-D=%s tokens=%d '
                'heads=%d temperature=%g.' % (
                    'on' if self.global_token_attention['hc_enabled'] else 'off',
                    'on' if self.global_token_attention['pd_enabled'] else 'off',
                    self.global_token_attention['tokens'],
                    self.global_token_attention['heads'],
                    self.global_token_attention['temperature'],
                )
            )
        if self.hplga['enabled']:
            if self.emb_size % self.hplga['heads'] != 0:
                raise ValueError(
                    'num.factors must be divisible by hplga.heads.'
                )
            if (
                not self.config.contains('attention.max.nodes')
                or int(self.config['attention.max.nodes']) != 0
            ):
                raise ValueError(
                    'HPLGA replaces dense full-node attention; set '
                    'attention.max.nodes=0.'
                )
            print(
                'HPLGA: H-C=%s P-D=%s heads=%d PageRank(alpha=%g, '
                'max_iter=%d, tol=%g) kernel=%s.' % (
                    'on' if self.hplga['hc_enabled'] else 'off',
                    'on' if self.hplga['pd_enabled'] else 'off',
                    self.hplga['heads'],
                    self.hplga['pagerank_alpha'],
                    self.hplga['pagerank_max_iter'],
                    self.hplga['pagerank_tol'],
                    self.hplga['kernel'],
                )
            )

        self.compound_cp_support_degrees = np.zeros(
            self.num_compounds, dtype=np.float32
        )
        self.protein_cp_support_degrees = np.zeros(
            self.num_proteins, dtype=np.float32
        )
        for compound_id, protein_id, label in self.data.cpassociation:
            if float(label) > 0:
                self.compound_cp_support_degrees[
                    self.data.compound[str(compound_id)]
                ] += 1.0
                self.protein_cp_support_degrees[
                    self.data.protein[str(protein_id)]
                ] += 1.0
        self.compound_context_available = (
            np.sum(self.compound_herb_mask, axis=1) > 0
        ).astype(np.float32)
        self.protein_context_available = (
            np.asarray(pd.sum(axis=1)).reshape(-1) > 0
        ).astype(np.float32)
        self.herb_prototypes = None
        self.jaccard_transfer = None
        if self.herb_prototype_transfer['enabled']:
            self.herb_prototypes = build_support_calibrated_herb_prototypes(
                hc, cp
            )
            if (
                self.herb_prototypes['training_positive_edges']
                != int(np.sum(self.compound_cp_support_degrees))
            ):
                raise ValueError(
                    'Herb prototype statistics disagree with strict training '
                    'C-P positive-edge counts.'
                )
            print(
                'Herb prototype source: %d strict training positives across '
                '%d supported compounds.' % (
                    self.herb_prototypes['training_positive_edges'],
                    self.herb_prototypes['supported_compound_count'],
                )
            )
            if self.herb_prototype_transfer['dual_granularity']:
                self.jaccard_transfer = build_loco_jaccard_transfer(hc, cp)
        self.hc_distillation_positive_pairs = None
        self.hc_distillation_positive_targets = None
        if self.hc_ranking_distillation['enabled']:
            cp_csr = cp.tocsr()
            positive_compounds, positive_proteins = cp_csr.nonzero()
            self.hc_distillation_positive_pairs = np.column_stack([
                positive_compounds, positive_proteins
            ]).astype(np.int64)
            self.hc_distillation_positive_targets = [
                cp_csr.indices[cp_csr.indptr[index]:cp_csr.indptr[index + 1]]
                for index in range(cp_csr.shape[0])
            ]
            print(
                'H-C ranking distillation: weight=%.3f temperature=%.3f '
                'queries=%d candidates=%d interval=%d.' % (
                    self.hc_ranking_distillation['weight'],
                    self.hc_ranking_distillation['temperature'],
                    self.hc_ranking_distillation['queries_per_update'],
                    self.hc_ranking_distillation['candidate_count'],
                    self.hc_ranking_distillation['interval'],
                )
            )
        self.safe_listwise_positive_targets = None
        self.safe_listwise_support_features = None
        self.safe_listwise_base_rank_cache = {}
        if self.safe_listwise_reranker['enabled']:
            cp_csr = cp.tocsr()
            self.safe_listwise_positive_targets = [
                cp_csr.indices[cp_csr.indptr[index]:cp_csr.indptr[index + 1]]
                for index in range(cp_csr.shape[0])
            ]
            feature_state = build_support_gate_features(hc, cp_csr)
            self.safe_listwise_support_features = feature_state['values']
            self.safe_listwise_feature_names = feature_state['feature_names']
            print(
                'Safe listwise reranker: base=%s tau=%.3f queries=%d '
                'hard=%d random=%d; outer test disabled for selection.' % (
                    self.safe_listwise_reranker['base_expert'],
                    self.safe_listwise_reranker['tau'],
                    self.safe_listwise_reranker['queries_per_update'],
                    self.safe_listwise_reranker['hard_unlabeled'],
                    self.safe_listwise_reranker['random_unlabeled'],
                )
            )
        if self.support_state_routing['enabled']:
            expected_state_codes = {
                'warm_warm': 0,
                'cold_warm': 1,
                'warm_cold': 2,
                'cold_cold': 3,
            }
            validation_state_counts = {}
            for state_name, expected_code in expected_state_codes.items():
                records = self.validationDataByState.get(state_name, [])
                if not records:
                    raise ValueError(
                        'Support-state routing is missing %s validation '
                        'records.' % state_name
                    )
                compound_indices = np.asarray([
                    self.data.compound[str(record[0])]
                    for record in records
                ], dtype=np.int64)
                protein_indices = np.asarray([
                    self.data.protein[str(record[1])]
                    for record in records
                ], dtype=np.int64)
                gates = support_state_pair_gates(
                    self.compound_cp_support_degrees[compound_indices],
                    self.protein_cp_support_degrees[protein_indices],
                    self.compound_context_available[compound_indices],
                    self.protein_context_available[protein_indices],
                )
                if np.any(gates['state'] != expected_code):
                    raise ValueError(
                        'Frozen %s records disagree with current training '
                        'C-P support degrees.' % state_name
                    )
                validation_state_counts[state_name] = len(records)
            print(
                'Support-state routing: WW=base+Hctx-P, CW=Hctx-P, '
                'WC=base, CC=Hctx-Dctx; validation=%s.' %
                ', '.join(
                    '%s:%d' % (name, validation_state_counts[name])
                    for name in (
                        'warm_warm', 'cold_warm',
                        'warm_cold', 'cold_cold',
                    )
                )
            )
        if (
            self.herb_context_attention['mode'] in TARGET_HERB_ATTENTION_MODES
            and not self.context_terms['herb_protein']
        ):
            raise ValueError(
                '%s requires context.herb_protein=True.' %
                self.herb_context_attention['mode']
            )
        if self.counterfactual_context['enabled']:
            if getattr(self.data, 'protocol', 'legacy') != 'strict':
                raise ValueError(
                    'Counterfactual herb context requires experiment.protocol=strict.'
                )
            if self.herb_context_attention['mode'] != 'static':
                raise ValueError(
                    'Counterfactual herb context requires static Hctx-P.'
                )
            if self.context_terms != {
                    'compound_disease': False,
                    'herb_protein': True,
                    'herb_disease': False}:
                raise ValueError(
                    'Counterfactual herb context requires HerbOnly context terms.'
                )
            from util.counterfactual_context import (
                build_exact_degree_counterfactuals,
            )
            compound_memberships = {
                str(self.data.id2compound[index]): {
                    str(self.data.id2herb[herb]) for herb in herbs
                }
                for index, herbs in enumerate(self.compound_herbs)
                if herbs
            }
            matching = build_exact_degree_counterfactuals(
                compound_memberships.keys(),
                compound_memberships,
                draws=self.counterfactual_context['draws'],
                seed=self.counterfactual_context['seed'],
            )
            donor_indices = np.tile(
                np.arange(self.num_compounds, dtype=np.int32)[:, None],
                (1, self.counterfactual_context['draws']),
            )
            donor_eligible = np.zeros(self.num_compounds, dtype=np.float32)
            for compound_id, donors in matching['assignments'].items():
                index = self.data.compound[str(compound_id)]
                donor_indices[index] = np.asarray([
                    self.data.compound[str(donor)] for donor in donors
                ], dtype=np.int32)
                donor_eligible[index] = 1.0
            self.counterfactual_donor_indices = donor_indices
            self.counterfactual_donor_eligible = donor_eligible
            self.counterfactual_u_idx = tf.placeholder(
                tf.int32, name='counterfactual_u_idx'
            )
            self.counterfactual_eligible_mask = tf.placeholder(
                tf.float32, name='counterfactual_eligible_mask'
            )
            import hashlib
            assignment_hash = hashlib.sha256(
                donor_indices.tobytes() + donor_eligible.tobytes()
            ).hexdigest()
            self.counterfactual_context['assignment_sha256'] = assignment_hash
            self.counterfactual_context['eligible_compounds'] = int(
                np.sum(donor_eligible)
            )
            print(
                'CHCR: enabled weight=%g margin=%g draws=%d seed=%d '
                'eligible_compounds=%d/%d assignment_hash=%s' % (
                    self.counterfactual_context['weight'],
                    self.counterfactual_context['margin'],
                    self.counterfactual_context['draws'],
                    self.counterfactual_context['seed'],
                    self.counterfactual_context['eligible_compounds'],
                    self.num_compounds,
                    assignment_hash[:12],
                )
            )
        else:
            self.counterfactual_donor_indices = None
            self.counterfactual_donor_eligible = None
            self.counterfactual_u_idx = None
            self.counterfactual_eligible_mask = None
        if self.context_mask_training['enabled']:
            if getattr(self.data, 'protocol', 'legacy') != 'strict':
                raise ValueError(
                    'Context-masked training requires experiment.protocol=strict.'
                )
            print(
                'CMIT: enabled side=%s weight=%g' % (
                    self.context_mask_training['side'],
                    self.context_mask_training['weight'],
                )
            )
        print(
            'Candidate context terms: C-Dctx=%s, Hctx-P=%s, Hctx-Dctx=%s' % (
                'on' if self.context_terms['compound_disease'] else 'off',
                'on' if self.context_terms['herb_protein'] else 'off',
                'on' if self.context_terms['herb_disease'] else 'off',
            )
        )
        print(
            'Herb context mode: %s (max incident herbs=%d)' % (
                self.herb_context_attention['mode'], self.max_compound_herbs
            )
        )
        if self.support_router['enabled']:
            print(
                'Support router: mode=%s pseudo_cold_ratio=%.4f seed=%d '
                'initial_slope=%g.' % (
                    self.support_router['mode'],
                    self.support_router['pseudo_cold_ratio'],
                    self.support_router['seed'],
                    self.support_router['initial_slope'],
                )
            )
        if self.support_experts['enabled']:
            print(
                'SCHE support-conditioned experts: mode=%s '
                'pseudo_cold_ratio=%.4f seed=%d detach_cold_features=%s.' % (
                    self.support_experts['mode'],
                    self.support_experts['pseudo_cold_ratio'],
                    self.support_experts['seed'],
                    'on' if self.support_experts[
                        'detach_cold_features'
                    ] else 'off',
                )
            )
        print('Pair decoder: %s' % self.pair_decoder['type'])
        if self.inductive_context['enabled']:
            print(
                'Support-decoupled inductive scoring: base_zero_support=%s '
                'self_excluded_hc=%s fallback_base_without_context=on.' % (
                    'on' if self.inductive_context[
                        'suppress_base_zero_support'
                    ] else 'off',
                    'on' if self.inductive_context['self_excluded'] else 'off',
                )
            )
        attention_size = 64
        self.compound_attention_weights = []
        self.protein_attention_weights = []

        num_heads = 2  # Number of attention heads
        head_dim = self.emb_size // num_heads  # Dimension of each attention head
        attention_max_nodes = None
        if self.config.contains('attention.max.nodes'):
            attention_max_nodes = int(self.config['attention.max.nodes'])
        use_compound_full_attention = (
            self.encoder_profile['use_dense_full_attention']
            and (
                attention_max_nodes is None
                or self.num_compounds <= attention_max_nodes
            )
        )
        use_protein_full_attention = (
            self.encoder_profile['use_dense_full_attention']
            and (
                attention_max_nodes is None
                or self.num_proteins <= attention_max_nodes
            )
        )
        global_attention_conflicts = (
            self.global_token_attention['hc_enabled']
            and use_compound_full_attention
        ) or (
            self.global_token_attention['pd_enabled']
            and use_protein_full_attention
        ) or (
            self.hplga['hc_enabled']
            and use_compound_full_attention
        ) or (
            self.hplga['pd_enabled']
            and use_protein_full_attention
        )
        if global_attention_conflicts:
            raise ValueError(
                'HILGA/HPLGA replaces dense full-node attention; set '
                'attention.max.nodes=0.'
            )
        if not self.encoder_profile['use_dense_full_attention']:
            print('Dense full-node self-attention disabled by encoder profile.')
        elif not use_compound_full_attention:
            print('Skipping compound full self-attention: nodes=%d > attention.max.nodes=%d' %
                  (self.num_compounds, attention_max_nodes))
        if (
            self.encoder_profile['use_dense_full_attention']
            and not use_protein_full_attention
        ):
            print('Skipping protein full self-attention: nodes=%d > attention.max.nodes=%d' %
                  (self.num_proteins, attention_max_nodes))



        pr_compound_embeddings = None
        pr_protein_embeddings = None
        if self.encoder_profile['use_pagerank']:
            if getattr(self.data, 'protocol', 'legacy') == 'strict':
                if not self.herb_prototype_transfer[
                        'replace_compound_pagerank']:
                    pr_compound_embeddings, _ = bipartite_pagerank(cp)
                pr_protein_embeddings, _ = bipartite_pagerank(pd)
                if self.herb_prototype_transfer[
                        'replace_compound_pagerank']:
                    print(
                        'Strict PageRank: compound C-P weighting replaced by '
                        'herb prototype transfer; protein P-D weighting retained.'
                    )
                else:
                    print(
                        'Strict PageRank: fold training C-P graph with '
                        'type-safe bipartite node IDs.'
                    )
            else:
                pr_compound = self.buildGraphAndPageRank(cp)
                pr_protein = self.buildGraphAndPageRank(pd)
                pr_compound_embeddings = np.array([
                    pr_compound.get(i, 0)
                    for i in range(self.num_compounds)
                ])
                pr_protein_embeddings = np.array([
                    pr_protein.get(i, 0)
                    for i in range(self.num_proteins)
                ])
            if pr_compound_embeddings is not None:
                pr_compound_embeddings = np.reshape(
                    pr_compound_embeddings, (self.num_compounds, 1)
                )
            pr_protein_embeddings = np.reshape(
                pr_protein_embeddings, (self.num_proteins, 1)
            )
        else:
            print('PageRank weighting disabled by encoder profile.')

        self.hplga_pagerank_diagnostics = {}
        hplga_priors = {}
        if self.hplga['hc_enabled']:
            hplga_priors['hc'], self.hplga_pagerank_diagnostics['hc'] = (
                hypergraph_pagerank(
                    H_c,
                    alpha=self.hplga['pagerank_alpha'],
                    max_iter=self.hplga['pagerank_max_iter'],
                    tol=self.hplga['pagerank_tol'],
                )
            )
        if self.hplga['pd_enabled']:
            hplga_priors['pd'], self.hplga_pagerank_diagnostics['pd'] = (
                hypergraph_pagerank(
                    pd,
                    alpha=self.hplga['pagerank_alpha'],
                    max_iter=self.hplga['pagerank_max_iter'],
                    tol=self.hplga['pagerank_tol'],
                )
            )
        hplga_prior_tensors = {
            side: tf.constant(
                prior, dtype=tf.float32, name=side + '_hplga_pagerank'
            )
            for side, prior in hplga_priors.items()
        }


        initializer = tf.variance_scaling_initializer(scale=2.0)

        for i in range(self.n_layer):
            self.weights['layer_%d' %(i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='JU_%d' % (i + 1))
            self.weights['layer_1_%d' %(i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='JU_1_%d' % (i + 1))
            if not self.encoder_profile['external_baseline']:
                self.weights['layer_2_%d' %(i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='JU_2_%d' % (i + 1))
                self.weights['layer_att_%d' %(i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='layer_bias_%d' %(i+1))
            if self.encoder_profile['use_node_dimension_attention']:
                self.attention_weights['compound'] = tf.Variable(
                    initializer([self.emb_size, self.emb_size]),
                    name='compound',
                )
                self.attention_weights['protein'] = tf.Variable(
                    initializer([self.emb_size, self.emb_size]),
                    name='protein',
                )
            for h in range(num_heads):
                # Keep the historical HDCTI variable graph unchanged even
                # when attention.max.nodes disables execution. The frozen
                # external baseline removes these variables entirely.
                if (
                    self.encoder_profile['use_dense_full_attention']
                    and not self.hplga['enabled']
                ):
                    self.attention_weights['compound_q_%d_%d' % (i + 1, h)] = tf.Variable(
                        initializer([self.emb_size, head_dim]), name='compound_q_%d_%d' % (i + 1, h))
                    self.attention_weights['compound_k_%d_%d' % (i + 1, h)] = tf.Variable(
                        initializer([self.emb_size, head_dim]), name='compound_k_%d_%d' % (i + 1, h))
                    self.attention_weights['compound_v_%d_%d' % (i + 1, h)] = tf.Variable(
                        initializer([self.emb_size, head_dim]), name='compound_v_%d_%d' % (i + 1, h))

                    self.attention_weights['protein_q_%d_%d' % (i + 1, h)] = tf.Variable(
                        initializer([self.emb_size, head_dim]), name='protein_q_%d_%d' % (i + 1, h))
                    self.attention_weights['protein_k_%d_%d' % (i + 1, h)] = tf.Variable(
                        initializer([self.emb_size, head_dim]), name='protein_k_%d_%d' % (i + 1, h))
                    self.attention_weights['protein_v_%d_%d' % (i + 1, h)] = tf.Variable(
                        initializer([self.emb_size, head_dim]), name='protein_v_%d_%d' % (i + 1, h))
            if self.hyperedge_attention['hc_enabled']:
                self.weights['hc_hyper_node_%d' % (i + 1)] = tf.Variable(
                    tf.zeros([self.emb_size]),
                    name='hc_hyper_node_%d' % (i + 1),
                )
                self.weights['hc_hyper_edge_%d' % (i + 1)] = tf.Variable(
                    tf.zeros([self.emb_size]),
                    name='hc_hyper_edge_%d' % (i + 1),
                )
            if self.hyperedge_attention['pd_enabled']:
                self.weights['pd_hyper_node_%d' % (i + 1)] = tf.Variable(
                    tf.zeros([self.emb_size]),
                    name='pd_hyper_node_%d' % (i + 1),
                )
                self.weights['pd_hyper_edge_%d' % (i + 1)] = tf.Variable(
                    tf.zeros([self.emb_size]),
                    name='pd_hyper_edge_%d' % (i + 1),
                )

        if self.encoder_profile['use_self_gating']:
            for i in range(2):
                self.weights['gating%d' % (i + 1)] = tf.Variable(
                    initializer([self.emb_size, self.emb_size]),
                    name='g_W_%d_1' % (i + 1),
                )
                self.weights['gating_bias%d' % (i + 1)] = tf.Variable(
                    initializer([1, self.emb_size]),
                    name='g_W_b_%d_1' % (i + 1),
                )

        if self.use_context_interaction:
            # Zero initialization keeps the first forward pass equal to the dot-product baseline.
            self.weights['context_compound_disease'] = tf.Variable(
                tf.zeros([self.emb_size]), name='context_compound_disease'
            )
            self.weights['context_herb_protein'] = tf.Variable(
                tf.zeros([self.emb_size]), name='context_herb_protein'
            )
            self.weights['context_herb_disease'] = tf.Variable(
                tf.zeros([self.emb_size]), name='context_herb_disease'
            )
            if self.support_experts['enabled']:
                self.weights['context_herb_protein_cold'] = tf.Variable(
                    tf.zeros([self.emb_size]),
                    name='context_herb_protein_cold',
                )
        if self.support_router['enabled']:
            self.weights['support_router_raw_slope'] = tf.Variable(
                np.asarray([
                    inverse_softplus(self.support_router['initial_slope'])
                ], dtype=np.float32),
                name='support_router_raw_slope',
            )
        if self.herb_context_attention['mode'] in TARGET_HERB_ATTENTION_MODES:
            self.weights['target_herb_projection'] = tf.Variable(
                tf.eye(self.emb_size, dtype=tf.float32),
                name='target_herb_projection',
            )
            self.weights['target_protein_projection'] = tf.Variable(
                tf.eye(self.emb_size, dtype=tf.float32),
                name='target_protein_projection',
            )
        if self.herb_context_attention['mode'] == 'target_residual_attention':
            # V2 starts exactly from static Hctx-P and learns only a pair-specific delta.
            self.weights['context_target_herb_residual'] = tf.Variable(
                tf.zeros([self.emb_size]),
                name='context_target_herb_residual',
            )
        if self.herb_prototype_transfer['enabled']:
            self.weights['herb_prototype_scale'] = tf.Variable(
                tf.zeros([1]), name='herb_prototype_scale',
                trainable=self.herb_prototype_transfer[
                    'prototype_component'
                ],
            )
            if (
                self.herb_prototype_transfer['dual_granularity']
                and self.herb_prototype_transfer['dual_mode']
                == 'independent_scales'
            ):
                self.weights['herb_jaccard_scale'] = tf.Variable(
                    tf.zeros([1]), name='herb_jaccard_scale'
                )

        if self.pair_decoder['type'] == 'bilinear':
            # Identity initialization makes the first forward pass equal to Dot.
            self.weights['decoder_bilinear'] = tf.Variable(
                tf.eye(self.emb_size, dtype=tf.float32), name='decoder_bilinear'
            )
        elif self.pair_decoder['type'] == 'mlp':
            hidden_size = self.pair_decoder['hidden_size']
            self.weights['decoder_mlp_hidden'] = tf.Variable(
                initializer([self.emb_size * 4, hidden_size]), name='decoder_mlp_hidden'
            )
            self.weights['decoder_mlp_hidden_bias'] = tf.Variable(
                tf.zeros([hidden_size]), name='decoder_mlp_hidden_bias'
            )
            # A zero output layer gives an exact Dot residual initialization.
            self.weights['decoder_mlp_output'] = tf.Variable(
                tf.zeros([hidden_size, 1]), name='decoder_mlp_output'
            )
            self.weights['decoder_mlp_output_bias'] = tf.Variable(
                tf.zeros([1]), name='decoder_mlp_output_bias'
            )

        if self.safe_listwise_reranker['enabled']:
            hidden_size = self.safe_listwise_reranker['hidden_size']
            gate_hidden_size = self.safe_listwise_reranker['gate_hidden_size']
            self.safe_listwise_weights['residual_hidden'] = tf.Variable(
                initializer([self.emb_size * 3, hidden_size]),
                name='safe_listwise_residual_hidden',
            )
            self.safe_listwise_weights['residual_hidden_bias'] = tf.Variable(
                tf.zeros([hidden_size]),
                name='safe_listwise_residual_hidden_bias',
            )
            self.safe_listwise_weights['residual_output'] = tf.Variable(
                initializer([hidden_size, 1]),
                name='safe_listwise_residual_output',
            )
            self.safe_listwise_weights['residual_output_bias'] = tf.Variable(
                tf.zeros([1]),
                name='safe_listwise_residual_output_bias',
            )
            self.safe_listwise_weights['gate_hidden'] = tf.Variable(
                initializer([4, gate_hidden_size]),
                name='safe_listwise_gate_hidden',
            )
            self.safe_listwise_weights['gate_hidden_bias'] = tf.Variable(
                tf.zeros([gate_hidden_size]),
                name='safe_listwise_gate_hidden_bias',
            )
            self.safe_listwise_weights['gate_output'] = tf.Variable(
                initializer([gate_hidden_size, 1]),
                name='safe_listwise_gate_output',
            )
            self.safe_listwise_weights['gate_output_bias'] = tf.Variable(
                tf.zeros([1]),
                name='safe_listwise_gate_output_bias',
            )
            # The initial correction is below 3e-6 at tau=0.05 while retaining
            # enough gradient for the safety scale to leave the fallback.
            self.safe_listwise_weights['gate_scale_raw'] = tf.Variable(
                tf.constant([-10.0], dtype=tf.float32),
                name='safe_listwise_gate_scale_raw',
            )

        self.global_token_diagnostic_tensors = {}
        if self.global_token_attention['enabled']:
            token_count = self.global_token_attention['tokens']
            for side in ('hc', 'pd'):
                if not self.global_token_attention[side + '_enabled']:
                    continue
                self.global_token_diagnostic_tensors[side] = []
                for layer in range(1, self.n_layer + 1):
                    prefix = '%s_global_token' % side
                    self.weights['%s_assignment_%d' % (prefix, layer)] = tf.Variable(
                        initializer([self.emb_size, token_count]),
                        name='%s_assignment_%d' % (prefix, layer),
                    )
                    for projection in ('q', 'k', 'v', 'output'):
                        self.weights['%s_%s_%d' % (
                            prefix, projection, layer
                        )] = tf.Variable(
                            initializer([self.emb_size, self.emb_size]),
                            name='%s_%s_%d' % (prefix, projection, layer),
                        )
                    self.weights['%s_gamma_%d' % (prefix, layer)] = tf.Variable(
                        tf.zeros([1]), name='%s_gamma_%d' % (prefix, layer)
                    )

        self.hplga_diagnostic_tensors = {}
        if self.hplga['enabled']:
            for side in ('hc', 'pd'):
                if not self.hplga[side + '_enabled']:
                    continue
                self.hplga_diagnostic_tensors[side] = []
                for layer in range(1, self.n_layer + 1):
                    prefix = '%s_hplga' % side
                    for projection in ('q', 'k', 'v', 'output'):
                        self.weights['%s_%s_%d' % (
                            prefix, projection, layer
                        )] = tf.Variable(
                            initializer([self.emb_size, self.emb_size]),
                            name='%s_%s_%d' % (
                                prefix, projection, layer
                            ),
                        )
                    self.weights['%s_gamma_%d' % (prefix, layer)] = (
                        tf.Variable(
                            tf.zeros([1]),
                            name='%s_gamma_%d' % (prefix, layer),
                        )
                    )

        def multi_head_attention_compound(embeddings, attention_weights, num_heads, head_dim):
            if not use_compound_full_attention:
                return embeddings
            attention_heads = []

            for h in range(num_heads):
                q = tf.matmul(embeddings, attention_weights['compound_q_%d_%d' % (i + 1, h)])
                k = tf.matmul(embeddings, attention_weights['compound_k_%d_%d' % (i + 1, h)])
                v = tf.matmul(embeddings, attention_weights['compound_v_%d_%d' % (i + 1, h)])

                attn_logits = tf.matmul(q, k, transpose_b=True)
                attn_weights = tf.nn.softmax(attn_logits / tf.sqrt(float(head_dim)), axis=-1)
                attn_output = tf.matmul(attn_weights, v)

                attention_heads.append(attn_output)

            # Concatenate all attention heads
            concat_attention = tf.concat(attention_heads, axis=-1)
            return concat_attention

        def multi_head_attention_protein(embeddings, attention_weights, num_heads, head_dim):
            if not use_protein_full_attention:
                return embeddings
            attention_heads = []

            for h in range(num_heads):
                q = tf.matmul(embeddings, attention_weights['protein_q_%d_%d' % (i + 1, h)])
                k = tf.matmul(embeddings, attention_weights['protein_k_%d_%d' % (i + 1, h)])
                v = tf.matmul(embeddings, attention_weights['protein_v_%d_%d' % (i + 1, h)])

                attn_logits = tf.matmul(q, k, transpose_b=True)
                attn_weights = tf.nn.softmax(attn_logits / tf.sqrt(float(head_dim)), axis=-1)
                attn_output = tf.matmul(attn_weights, v)

                attention_heads.append(attn_output)

            # Concatenate all attention heads
            concat_attention = tf.concat(attention_heads, axis=-1)
            return concat_attention


        def self_gating(em,channel):

            return tf.multiply(em,tf.nn.sigmoid(tf.matmul(em,self.weights['gating%d' % channel])+self.weights['gating_bias%d' %channel]))

        def unsorted_segment_softmax(logits, segment_ids, segment_count):
            maxima = tf.math.unsorted_segment_max(
                logits, segment_ids, segment_count
            )
            shifted = logits - tf.gather(maxima, segment_ids)
            exponentials = tf.exp(shifted)
            denominators = tf.math.unsorted_segment_sum(
                exponentials, segment_ids, segment_count
            )
            return exponentials / (
                tf.gather(denominators, segment_ids) + 1e-12
            )

        def factorized_hyperedge_propagation(
                node_embeddings,
                forward_edge_ids,
                forward_node_ids,
                reverse_edge_ids,
                reverse_node_ids,
                edge_count,
                node_count,
                specificity_prior,
                node_score_vector,
                edge_score_vector,
                name):
            temperature = float(self.hyperedge_attention['temperature'])
            prior_scale = float(self.hyperedge_attention['prior_scale'])
            node_logits = tf.reduce_sum(
                node_embeddings * node_score_vector, axis=1
            )
            node_to_edge_values = unsorted_segment_softmax(
                tf.gather(node_logits, forward_node_ids) / temperature,
                forward_edge_ids,
                edge_count,
            )
            node_to_edge = tf.SparseTensor(
                tf.stack([forward_edge_ids, forward_node_ids], axis=1),
                node_to_edge_values,
                [edge_count, node_count],
            )
            edge_embeddings = tf.sparse_tensor_dense_matmul(
                node_to_edge,
                node_embeddings,
                name=name + '_node_to_edge',
            )
            edge_logits = tf.reduce_sum(
                edge_embeddings * edge_score_vector, axis=1
            ) + prior_scale * specificity_prior
            edge_to_node_values = unsorted_segment_softmax(
                tf.gather(edge_logits, reverse_edge_ids) / temperature,
                reverse_node_ids,
                node_count,
            )
            edge_to_node = tf.SparseTensor(
                tf.stack([reverse_node_ids, reverse_edge_ids], axis=1),
                edge_to_node_values,
                [node_count, edge_count],
            )
            propagated_nodes = tf.sparse_tensor_dense_matmul(
                edge_to_node,
                edge_embeddings,
                name=name + '_edge_to_node',
            )
            return edge_embeddings, propagated_nodes

        def hyperedge_induced_global_attention(
                node_embeddings,
                edge_embeddings,
                active_edge_mask,
                side,
                layer):
            token_count = self.global_token_attention['tokens']
            head_count = self.global_token_attention['heads']
            head_dim = self.emb_size // head_count
            temperature = self.global_token_attention['temperature']
            prefix = '%s_global_token' % side

            normalized_edges = tf.math.l2_normalize(edge_embeddings, axis=1)
            assignment_logits = tf.matmul(
                normalized_edges,
                self.weights['%s_assignment_%d' % (prefix, layer)],
            )
            edge_mask = tf.reshape(active_edge_mask, [-1, 1])
            assignment_logits += (1.0 - edge_mask) * -1e9
            edge_assignments = tf.nn.softmax(
                assignment_logits / temperature,
                axis=0,
                name='%s_assignment_layer_%d' % (prefix, layer),
            )
            token_embeddings = tf.matmul(
                edge_assignments,
                edge_embeddings,
                transpose_a=True,
                name='%s_pool_layer_%d' % (prefix, layer),
            )

            queries = tf.matmul(
                tf.math.l2_normalize(node_embeddings, axis=1),
                self.weights['%s_q_%d' % (prefix, layer)],
            )
            keys = tf.matmul(
                tf.math.l2_normalize(token_embeddings, axis=1),
                self.weights['%s_k_%d' % (prefix, layer)],
            )
            values = tf.matmul(
                token_embeddings,
                self.weights['%s_v_%d' % (prefix, layer)],
            )
            queries = tf.transpose(
                tf.reshape(queries, [-1, head_count, head_dim]), [1, 0, 2]
            )
            keys = tf.transpose(
                tf.reshape(keys, [token_count, head_count, head_dim]), [1, 0, 2]
            )
            values = tf.transpose(
                tf.reshape(values, [token_count, head_count, head_dim]), [1, 0, 2]
            )
            node_attention = tf.nn.softmax(
                tf.matmul(queries, keys, transpose_b=True)
                / (tf.sqrt(float(head_dim)) * temperature),
                axis=-1,
                name='%s_node_attention_layer_%d' % (prefix, layer),
            )
            attended = tf.matmul(node_attention, values)
            attended = tf.reshape(
                tf.transpose(attended, [1, 0, 2]), [-1, self.emb_size]
            )
            update = tf.matmul(
                attended,
                self.weights['%s_output_%d' % (prefix, layer)],
                name='%s_update_layer_%d' % (prefix, layer),
            )
            residual_scale = tf.tanh(
                self.weights['%s_gamma_%d' % (prefix, layer)][0]
            )
            self.global_token_diagnostic_tensors[side].append({
                'edge_assignments': edge_assignments,
                'node_attention': node_attention,
                'token_embeddings': token_embeddings,
            })
            return tf.add(
                node_embeddings,
                residual_scale * update,
                name='%s_residual_layer_%d' % (prefix, layer),
            )

        def apply_hplga(node_embeddings, side, layer):
            prefix = '%s_hplga' % side
            attention_parameters = {
                projection: self.weights['%s_%s_%d' % (
                    prefix, projection, layer
                )]
                for projection in ('q', 'k', 'v', 'output')
            }
            attention_parameters['gamma'] = self.weights[
                '%s_gamma_%d' % (prefix, layer)
            ]
            output, diagnostics = pagerank_linear_attention_tf(
                tf,
                node_embeddings,
                hplga_prior_tensors[side],
                attention_parameters,
                self.hplga['heads'],
                self.hplga['epsilon'],
                '%s_layer_%d' % (prefix, layer),
            )
            self.hplga_diagnostic_tensors[side].append(diagnostics)
            return output

        hc_attention_edge_ids_tf = tf.constant(
            hc_attention_ids['forward_edge_ids'], dtype=tf.int64
        )
        hc_attention_node_ids_tf = tf.constant(
            hc_attention_ids['forward_node_ids'], dtype=tf.int64
        )
        hc_attention_reverse_edge_ids_tf = tf.constant(
            hc_attention_ids['reverse_edge_ids'], dtype=tf.int64
        )
        hc_attention_reverse_node_ids_tf = tf.constant(
            hc_attention_ids['reverse_node_ids'], dtype=tf.int64
        )
        hc_specificity_prior_tf = tf.constant(
            hc_specificity_prior, dtype=tf.float32
        )
        hc_active_edge_mask_tf = tf.constant(
            (hc_edge_degrees > 0).astype(np.float32), dtype=tf.float32
        )
        hc_edge_degrees_tf = tf.constant(
            hc_edge_degrees, dtype=tf.float32
        )
        pd_attention_edge_ids_tf = tf.constant(
            pd_attention_ids['forward_edge_ids'], dtype=tf.int64
        )
        pd_attention_node_ids_tf = tf.constant(
            pd_attention_ids['forward_node_ids'], dtype=tf.int64
        )
        pd_attention_reverse_edge_ids_tf = tf.constant(
            pd_attention_ids['reverse_edge_ids'], dtype=tf.int64
        )
        pd_attention_reverse_node_ids_tf = tf.constant(
            pd_attention_ids['reverse_node_ids'], dtype=tf.int64
        )
        pd_specificity_prior_tf = tf.constant(
            pd_specificity_prior, dtype=tf.float32
        )
        pd_active_edge_mask_tf = tf.constant(
            (pd_edge_degrees > 0).astype(np.float32), dtype=tf.float32
        )

        if self.encoder_profile['use_self_gating']:
            compound_embeddings = self_gating(self.compound_embeddings, 1)
            protein_embeddings = self_gating(self.protein_embeddings, 2)
        else:
            compound_embeddings = self.compound_embeddings
            protein_embeddings = self.protein_embeddings

        all_compound_embeddings = [compound_embeddings]

        protein_embeddings = protein_embeddings
        all_protein_embeddings = [protein_embeddings]
        all_hc_embeddings = []
        all_pd_embeddings = []
        all_hc_edge_inputs = []
        all_self_excluded_hc_contexts = []

        def direct_self_excluded_hc_context(
                node_embeddings, edge_embeddings, layer):
            incidence_degrees = tf.gather(
                hc_edge_degrees_tf, hc_attention_edge_ids_tf
            )
            incidence_nodes = tf.gather(
                node_embeddings, hc_attention_node_ids_tf
            )
            incidence_edges = tf.gather(
                edge_embeddings, hc_attention_edge_ids_tf
            )
            denominators = tf.maximum(incidence_degrees - 1.0, 1.0)
            excluded_edges = (
                incidence_edges * tf.expand_dims(incidence_degrees, axis=1)
                - incidence_nodes
            ) / tf.expand_dims(denominators, axis=1)
            eligible = tf.cast(incidence_degrees > 1.0, tf.float32)
            excluded_edges = tf.math.l2_normalize(
                excluded_edges, axis=1
            ) * tf.expand_dims(eligible, axis=1)
            return tf.math.unsorted_segment_sum(
                excluded_edges,
                hc_attention_node_ids_tf,
                self.num_compounds,
                name='hc_self_excluded_context_layer_%d' % layer,
            )

        for i in range(self.n_layer):
            # Retain the exact node-side inputs used by H_e at each layer. The
            # stacked tensor is diagnostic-only and introduces no new variable.
            all_hc_edge_inputs.append(compound_embeddings)
            if self.hyperedge_attention['hc_enabled']:
                new_hc_edge, new_compound_embeddings = (
                    factorized_hyperedge_propagation(
                        compound_embeddings,
                        hc_attention_edge_ids_tf,
                        hc_attention_node_ids_tf,
                        hc_attention_reverse_edge_ids_tf,
                        hc_attention_reverse_node_ids_tf,
                        self.num_herbs,
                        self.num_compounds,
                        hc_specificity_prior_tf,
                        self.weights['hc_hyper_node_%d' % (i + 1)],
                        self.weights['hc_hyper_edge_%d' % (i + 1)],
                        'hc_factorized_attention_layer_%d' % (i + 1),
                    )
                )
            else:
                new_hc_edge = tf.sparse_tensor_dense_matmul(
                    H_e,
                    compound_embeddings,
                    name='hc_node_to_edge_layer_%d' % (i + 1),
                )
                new_compound_embeddings = tf.sparse_tensor_dense_matmul(
                    H_n,
                    new_hc_edge,
                    name='hc_edge_to_node_layer_%d' % (i + 1),
                )
            if self.inductive_context['self_excluded']:
                all_self_excluded_hc_contexts.append(
                    direct_self_excluded_hc_context(
                        compound_embeddings, new_hc_edge, i + 1
                    )
                )
            if self.hyperedge_attention['pd_enabled']:
                new_pd_edge, new_protein_embeddings = (
                    factorized_hyperedge_propagation(
                        protein_embeddings,
                        pd_attention_edge_ids_tf,
                        pd_attention_node_ids_tf,
                        pd_attention_reverse_edge_ids_tf,
                        pd_attention_reverse_node_ids_tf,
                        self.num_diseases,
                        self.num_proteins,
                        pd_specificity_prior_tf,
                        self.weights['pd_hyper_node_%d' % (i + 1)],
                        self.weights['pd_hyper_edge_%d' % (i + 1)],
                        'pd_factorized_attention_layer_%d' % (i + 1),
                    )
                )
            else:
                new_pd_edge = tf.sparse_tensor_dense_matmul(
                    P_e,
                    protein_embeddings,
                    name='pd_node_to_edge_layer_%d' % (i + 1),
                )
                new_protein_embeddings = tf.sparse_tensor_dense_matmul(
                    P_n,
                    new_pd_edge,
                    name='pd_edge_to_node_layer_%d' % (i + 1),
                )
            if self.encoder_profile['use_pagerank']:
                if pr_compound_embeddings is not None:
                    new_compound_embeddings = (
                        new_compound_embeddings * pr_compound_embeddings
                    )
                new_protein_embeddings = (
                    new_protein_embeddings * pr_protein_embeddings
                )
        
            new_compound_embeddings = multi_head_attention_compound(new_compound_embeddings, self.attention_weights, num_heads,
                                                            head_dim)
            new_compound_embeddings = tf.nn.leaky_relu(
                tf.matmul(new_compound_embeddings, self.weights['layer_%d' % (i + 1)]) + compound_embeddings)
            if self.hplga['hc_enabled']:
                new_compound_embeddings = apply_hplga(
                    new_compound_embeddings, 'hc', i + 1
                )
            if self.global_token_attention['hc_enabled']:
                new_compound_embeddings = hyperedge_induced_global_attention(
                    new_compound_embeddings,
                    new_hc_edge,
                    hc_active_edge_mask_tf,
                    'hc',
                    i + 1,
                )
        
            # 计算蛋白质节点的新嵌入
            new_protein_embeddings = multi_head_attention_protein(new_protein_embeddings, self.attention_weights, num_heads,
                                                          head_dim)
            new_protein_embeddings = tf.nn.leaky_relu(
                tf.matmul(new_protein_embeddings, self.weights['layer_1_%d' % (i + 1)]) + protein_embeddings)
            if self.hplga['pd_enabled']:
                new_protein_embeddings = apply_hplga(
                    new_protein_embeddings, 'pd', i + 1
                )
            if self.global_token_attention['pd_enabled']:
                new_protein_embeddings = hyperedge_induced_global_attention(
                    new_protein_embeddings,
                    new_pd_edge,
                    pd_active_edge_mask_tf,
                    'pd',
                    i + 1,
                )
        
            if self.encoder_profile['use_node_dimension_attention']:
                attn_weights_compound = tf.nn.softmax(tf.matmul(
                    new_compound_embeddings,
                    self.attention_weights['compound'],
                ))
                attn_weights_protein = tf.nn.softmax(tf.matmul(
                    new_protein_embeddings,
                    self.attention_weights['protein'],
                ))
                new_compound_embeddings = tf.nn.leaky_relu(tf.matmul(
                    attn_weights_compound * new_compound_embeddings,
                    self.weights['layer_%d' % (i + 1)],
                ) + compound_embeddings)
                new_protein_embeddings = tf.nn.leaky_relu(tf.matmul(
                    attn_weights_protein * new_protein_embeddings,
                    self.weights['layer_1_%d' % (i + 1)],
                ) + protein_embeddings)
        
            compound_embeddings = tf.nn.leaky_relu(new_compound_embeddings)
            protein_embeddings = tf.nn.leaky_relu(new_protein_embeddings)
            # compound_embeddings = tf.nn.leaky_relu(compound_embeddings)
            # protein_embeddings = tf.nn.leaky_relu(protein_embeddings)
        
            compound_embeddings = tf.math.l2_normalize(compound_embeddings,axis=1)
            protein_embeddings = tf.math.l2_normalize(protein_embeddings,axis=1)
            new_hc_edge=tf.math.l2_normalize(new_hc_edge,axis=1)
            new_pd_edge=tf.math.l2_normalize(new_pd_edge,axis=1)
        
        
        
            all_compound_embeddings+=[compound_embeddings]
            all_protein_embeddings+=[protein_embeddings]
            all_hc_embeddings+=[new_hc_edge]
            all_pd_embeddings+=[new_pd_edge]


        compound_embeddings = tf.reduce_sum(all_compound_embeddings,axis=0)
        protein_embeddings = tf.reduce_sum(all_protein_embeddings, axis=0)
        # compound_embeddings = tf.math.l2_normalize(compound_embeddings)
        # protein_embeddings = tf.math.l2_normalize(protein_embeddings)
        compound_embeddings = tf.nn.leaky_relu(compound_embeddings, alpha=0.2)
        protein_embeddings = tf.nn.leaky_relu(protein_embeddings,alpha=0.2)
        # a = tf.reduce_sum(all_compound_embeddings, axis=0)
        # b = tf.reduce_sum(all_protein_embeddings, axis=0)
        hc_edge=tf.reduce_sum(all_hc_embeddings,axis=0)
        pd_edge=tf.reduce_sum(all_pd_embeddings,axis=0)

        # new_hc_dege = tf.nn.leaky_relu(hc_edge, alpha=0.2)
        # new_pd_edge = tf.nn.leaky_relu(pd_edge, alpha=0.2)



        self.neg_idx = tf.placeholder(tf.float32, name="neg_holder")

        self.neg_disease_embedding = tf.convert_to_tensor(self.neg_idx,dtype=tf.float32)
        #self.neg_disease_embedding = tf.nn.embedding_lookup(tf.convert_to_tensor(A.toarray(),dtype=tf.float32), self.u_idx)

        self.final_iembedding = protein_embeddings
        self.final_uembedding = compound_embeddings

        self.final_hcedge=hc_edge
        self.final_pdedge=pd_edge
        self.final_hc_edge_inputs = tf.stack(
            all_hc_edge_inputs, axis=0, name='hc_edge_layer_inputs'
        )

        # Aggregate only the candidate's incident H-C/P-D hyperedges. No H-D or C-P labels are used here.
        self.final_compound_context = tf.math.l2_normalize(
            tf.sparse_tensor_dense_matmul(
                H_n, self.final_hcedge, name='hc_candidate_context'
            ),
            axis=1,
        )
        self.final_self_excluded_compound_context = None
        if self.inductive_context['self_excluded']:
            self.final_self_excluded_compound_context = tf.math.l2_normalize(
                tf.reduce_sum(all_self_excluded_hc_contexts, axis=0),
                axis=1,
                name='hc_self_excluded_candidate_context',
            )
        self.final_scoring_compound_context = (
            self.final_self_excluded_compound_context
            if self.inductive_context['self_excluded']
            else self.final_compound_context
        )
        self.final_protein_context = tf.math.l2_normalize(
            tf.sparse_tensor_dense_matmul(
                P_n, self.final_pdedge, name='pd_candidate_context'
            ),
            axis=1,
        )

        # self.u_embedding = tf.nn.embedding_lookup(self.final_hcedge, self.u_idx)
        # self.v_embedding = tf.nn.embedding_lookup(self.final_pdedge, self.v_idx)
        self.u_embedding = tf.nn.embedding_lookup(self.final_uembedding, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.final_iembedding, self.v_idx)
        self.u_context_embedding = tf.nn.embedding_lookup(
            self.final_scoring_compound_context, self.u_idx
        )
        self.v_context_embedding = tf.nn.embedding_lookup(self.final_protein_context, self.v_idx)
        self.support_state_pair_gates = None
        if self.support_state_routing['enabled']:
            pair_compound_support = tf.gather(
                tf.constant(
                    self.compound_cp_support_degrees, dtype=tf.float32
                ),
                self.u_idx,
            )
            pair_protein_support = tf.gather(
                tf.constant(
                    self.protein_cp_support_degrees, dtype=tf.float32
                ),
                self.v_idx,
            )
            pair_compound_context = tf.gather(
                tf.constant(
                    self.compound_context_available, dtype=tf.float32
                ),
                self.u_idx,
            )
            pair_protein_context = tf.gather(
                tf.constant(
                    self.protein_context_available, dtype=tf.float32
                ),
                self.v_idx,
            )
            compound_warm = pair_compound_support > 0
            protein_warm = pair_protein_support > 0
            self.support_state_pair_gates = {
                'base': tf.cast(compound_warm, tf.float32),
                'herb_protein': tf.cast(
                    tf.logical_and(
                        protein_warm, pair_compound_context > 0
                    ),
                    tf.float32,
                ),
                'herb_disease': tf.cast(
                    tf.logical_and(
                        tf.logical_and(
                            tf.logical_not(compound_warm),
                            tf.logical_not(protein_warm),
                        ),
                        tf.logical_and(
                            pair_compound_context > 0,
                            pair_protein_context > 0,
                        ),
                    ),
                    tf.float32,
                ),
            }
        self.inductive_base_gate = None
        if self.inductive_context['suppress_base_zero_support']:
            pair_support_degrees = tf.gather(
                tf.constant(
                    self.compound_cp_support_degrees, dtype=tf.float32
                ),
                self.u_idx,
            )
            pair_inductive_context_available = tf.gather(
                tf.constant(
                    self.compound_inductive_context_available,
                    dtype=tf.float32,
                ),
                self.u_idx,
            )
            suppress_base = tf.cast(
                tf.logical_and(
                    pair_support_degrees <= 0,
                    pair_inductive_context_available > 0,
                ),
                tf.float32,
            )
            self.inductive_base_gate = 1.0 - suppress_base
        self.support_context_gate = None
        if self.support_router['enabled']:
            pair_support_degrees = tf.gather(
                tf.constant(
                    self.compound_cp_support_degrees, dtype=tf.float32
                ),
                self.u_idx,
            )
            pair_context_available = tf.gather(
                tf.constant(
                    self.compound_context_available, dtype=tf.float32
                ),
                self.u_idx,
            )
            support_slope = tf.nn.softplus(
                self.weights['support_router_raw_slope'][0]
            )
            self.support_context_gate = pair_context_available * tf.exp(
                -support_slope * tf.log1p(pair_support_degrees)
            )
        self.support_expert_cold_gate = None
        if self.support_experts['enabled']:
            pair_support_degrees = tf.gather(
                tf.constant(
                    self.compound_cp_support_degrees, dtype=tf.float32
                ),
                self.u_idx,
            )
            pair_context_available = tf.gather(
                tf.constant(
                    self.compound_context_available, dtype=tf.float32
                ),
                self.u_idx,
            )
            self.support_expert_cold_gate = tf.cast(
                tf.logical_and(
                    pair_support_degrees <= 0,
                    pair_context_available > 0,
                ),
                tf.float32,
                name='support_expert_cold_gate',
            )
        self.target_herb_attention_weights = None
        self.target_herb_context_embedding = None
        if self.herb_context_attention['mode'] in TARGET_HERB_ATTENTION_MODES:
            padded_herb_edges = tf.concat(
                [
                    self.final_hcedge,
                    tf.zeros([1, self.emb_size], dtype=tf.float32),
                ],
                axis=0,
            )
            pair_herb_indices = tf.gather(
                tf.constant(self.compound_herb_indices, dtype=tf.int32),
                self.u_idx,
            )
            pair_herb_mask = tf.gather(
                tf.constant(self.compound_herb_mask, dtype=tf.float32),
                self.u_idx,
            )
            pair_herb_edges = tf.gather(padded_herb_edges, pair_herb_indices)
            pair_herb_edges.set_shape(
                [None, self.max_compound_herbs, self.emb_size]
            )
            herb_keys = tf.tensordot(
                pair_herb_edges,
                self.weights['target_herb_projection'],
                axes=[[2], [0]],
            )
            herb_keys.set_shape([None, self.max_compound_herbs, self.emb_size])
            protein_queries = tf.matmul(
                self.v_embedding, self.weights['target_protein_projection']
            )
            attention_logits = tf.reduce_sum(
                herb_keys * tf.expand_dims(protein_queries, axis=1), axis=2
            )
            attention_logits.set_shape([None, self.max_compound_herbs])
            attention_logits /= (
                tf.sqrt(tf.cast(self.emb_size, tf.float32))
                * self.herb_context_attention['temperature']
            )
            masked_logits = tf.where(
                pair_herb_mask > 0,
                attention_logits,
                tf.fill(tf.shape(attention_logits), tf.constant(-1e9, tf.float32)),
            )
            masked_logits = tf.reshape(
                masked_logits, [-1, self.max_compound_herbs]
            )
            attention_weights = tf.nn.softmax(masked_logits, axis=-1) * pair_herb_mask
            attention_weights /= (
                tf.reduce_sum(attention_weights, axis=1, keepdims=True) + 1e-12
            )
            self.target_herb_attention_weights = attention_weights
            self.target_herb_context_embedding = tf.math.l2_normalize(
                tf.reduce_sum(
                    tf.expand_dims(attention_weights, axis=2) * pair_herb_edges,
                    axis=1,
                ),
                axis=1,
            )
            if self.herb_context_attention['mode'] == 'target_attention':
                self.u_context_embedding = self.target_herb_context_embedding

        self.herb_prototype_pair_residual = None
        self.herb_jaccard_pair_residual = None
        if self.herb_prototype_transfer['enabled']:
            prototype_residual = self.buildHerbPrototypePairResidual()
            if self.herb_prototype_transfer['dual_granularity']:
                jaccard_residual = self.buildJaccardPairResidual()
                if (
                    self.herb_prototype_transfer['dual_mode']
                    == 'independent_scales'
                ):
                    self.herb_prototype_pair_residual = prototype_residual
                    self.herb_jaccard_pair_residual = jaccard_residual
                else:
                    blend = self.herb_prototype_transfer['jaccard_blend']
                    self.herb_prototype_pair_residual = (
                        (1.0 - blend) * prototype_residual
                        + blend * jaccard_residual
                    )
            else:
                self.herb_prototype_pair_residual = prototype_residual

        #self.v_embedding = self.final_pdedge


    def buildHerbPrototypePairResidual(self):
        prototypes = self.herb_prototypes
        target_counts = tf.constant(
            np.vstack([
                prototypes['herb_target_counts'],
                np.zeros((1, self.num_proteins), dtype=np.float32),
            ]),
            dtype=tf.float32,
        )
        support_counts = tf.constant(
            np.concatenate([
                prototypes['herb_support_counts'],
                np.zeros(1, dtype=np.float32),
            ]),
            dtype=tf.float32,
        )
        pair_herbs = tf.gather(
            tf.constant(self.compound_herb_indices, dtype=tf.int32),
            self.u_idx,
        )
        pair_mask = tf.gather(
            tf.constant(self.compound_herb_mask, dtype=tf.float32),
            self.u_idx,
        )
        pair_targets = tf.tile(
            tf.expand_dims(self.v_idx, axis=1),
            [1, self.max_compound_herbs],
        )
        herb_target_indices = tf.stack(
            [pair_herbs, pair_targets], axis=2
        )
        pair_target_counts = tf.gather_nd(
            target_counts, herb_target_indices
        )
        pair_support_counts = tf.gather(support_counts, pair_herbs)
        pair_compound_supported = tf.gather(
            tf.constant(
                prototypes['compound_supported'], dtype=tf.float32
            ),
            self.u_idx,
        )
        training_edge_keys = np.asarray(
            prototypes['training_edge_keys'], dtype=np.int64
        )
        pair_keys = (
            tf.cast(self.u_idx, tf.int64)
            * tf.constant(self.num_proteins, dtype=tf.int64)
            + tf.cast(self.v_idx, tf.int64)
        )
        if training_edge_keys.size:
            edge_keys = tf.constant(training_edge_keys, dtype=tf.int64)
            edge_positions = tf.searchsorted(
                edge_keys, pair_keys, side='left', out_type=tf.int32
            )
            safe_positions = tf.minimum(
                edge_positions,
                tf.constant(training_edge_keys.size - 1, dtype=tf.int32),
            )
            pair_self_positive = tf.cast(
                tf.logical_and(
                    edge_positions < training_edge_keys.size,
                    tf.equal(
                        tf.gather(edge_keys, safe_positions), pair_keys
                    ),
                ),
                tf.float32,
            )
        else:
            pair_self_positive = tf.zeros_like(
                self.u_idx, dtype=tf.float32
            )
        source_counts = (
            pair_support_counts
            - tf.expand_dims(pair_compound_supported, axis=1)
        )
        pair_target_counts -= tf.expand_dims(
            pair_self_positive, axis=1
        )
        target_prior = tf.expand_dims(
            tf.gather(
                tf.constant(
                    prototypes['target_prevalence'], dtype=tf.float32
                ),
                self.v_idx,
            ),
            axis=1,
        )
        prior_strength = tf.constant(
            self.herb_prototype_transfer['prior_strength'],
            dtype=tf.float32,
        )
        valid = pair_mask * tf.cast(source_counts > 0, tf.float32)
        posterior = (
            pair_target_counts + prior_strength * target_prior
        ) / tf.maximum(source_counts + prior_strength, prior_strength)
        residual = (posterior - target_prior) * valid
        valid_count = tf.reduce_sum(valid, axis=1)
        return tf.math.divide_no_nan(
            tf.reduce_sum(residual, axis=1), valid_count,
            name='support_calibrated_loco_herb_prototype_residual',
        )

    def herbPrototypePairResiduals(
            self, compound_indices, protein_indices,
            return_diagnostics=False):
        if not self.herb_prototype_transfer['enabled']:
            return None
        prototype_residual = support_calibrated_herb_prototype_scores(
            self.herb_prototypes,
            self.compound_herb_indices,
            self.compound_herb_mask,
            compound_indices,
            protein_indices,
            prior_strength=self.herb_prototype_transfer['prior_strength'],
            return_diagnostics=return_diagnostics,
        )
        if not self.herb_prototype_transfer['dual_granularity']:
            return prototype_residual
        if (
            self.herb_prototype_transfer['dual_mode']
            == 'independent_scales'
        ):
            return prototype_residual
        if return_diagnostics:
            prototype_values, prototype_diagnostics = prototype_residual
            jaccard_values, jaccard_diagnostics = (
                loco_jaccard_pair_residuals(
                    self.jaccard_transfer,
                    compound_indices,
                    protein_indices,
                    return_diagnostics=True,
                )
            )
            blend = self.herb_prototype_transfer['jaccard_blend']
            values = (
                (1.0 - blend) * prototype_values
                + blend * jaccard_values
            )
            return values.astype(np.float32), {
                'pairs': prototype_diagnostics['pairs'],
                'evidence_pairs': int(np.sum(
                    (np.abs(prototype_values) > 0)
                    | (np.abs(jaccard_values) > 0)
                )),
                'evidence_coverage': float(np.mean(
                    (np.abs(prototype_values) > 0)
                    | (np.abs(jaccard_values) > 0)
                )),
                'mean_abs_residual': float(np.mean(np.abs(values))),
                'prototype': prototype_diagnostics,
                'jaccard': jaccard_diagnostics,
            }
        jaccard_residual = loco_jaccard_pair_residuals(
            self.jaccard_transfer, compound_indices, protein_indices
        )
        blend = self.herb_prototype_transfer['jaccard_blend']
        return (
            (1.0 - blend) * prototype_residual
            + blend * jaccard_residual
        ).astype(np.float32)

    def herbJaccardPairResiduals(
            self, compound_indices, protein_indices,
            return_diagnostics=False):
        if not (
            self.herb_prototype_transfer['enabled']
            and self.herb_prototype_transfer['dual_granularity']
        ):
            return None
        return loco_jaccard_pair_residuals(
            self.jaccard_transfer,
            compound_indices,
            protein_indices,
            return_diagnostics=return_diagnostics,
        )

    def buildJaccardPairResidual(self):
        pair_records = list(self.data.trainingData)
        pair_records.extend(getattr(self, 'validationData', []) or [])
        pair_records.extend(getattr(self.data, 'testData', []) or [])
        training_compounds = np.asarray([
            self.data.compound[str(record[0])]
            for record in pair_records
        ], dtype=np.int64)
        training_proteins = np.asarray([
            self.data.protein[str(record[1])]
            for record in pair_records
        ], dtype=np.int64)
        training_values = loco_jaccard_pair_residuals(
            self.jaccard_transfer, training_compounds, training_proteins
        )
        training_keys = (
            training_compounds * np.int64(self.num_proteins)
            + training_proteins
        )
        order = np.argsort(training_keys, kind='mergesort')
        sorted_keys = training_keys[order]
        sorted_values = training_values[order]
        unique_keys, unique_positions = np.unique(
            sorted_keys, return_index=True
        )
        unique_values = sorted_values[unique_positions]

        pair_keys = (
            tf.cast(self.u_idx, tf.int64)
            * tf.constant(self.num_proteins, dtype=tf.int64)
            + tf.cast(self.v_idx, tf.int64)
        )
        keys = tf.constant(unique_keys, dtype=tf.int64)
        values = tf.constant(unique_values, dtype=tf.float32)
        positions = tf.searchsorted(
            keys, pair_keys, side='left', out_type=tf.int32
        )
        safe_positions = tf.minimum(
            positions, tf.constant(unique_keys.size - 1, dtype=tf.int32)
        )
        found = tf.logical_and(
            positions < unique_keys.size,
            tf.equal(tf.gather(keys, safe_positions), pair_keys),
        )
        return tf.where(
            found,
            tf.gather(values, safe_positions),
            tf.zeros_like(tf.cast(self.u_idx, tf.float32)),
            name='loco_hc_jaccard_transfer_residual',
        )


    def buildBasePairLogits(self, compound_embedding=None, protein_embedding=None):
        compound_embedding = (
            self.u_embedding if compound_embedding is None else compound_embedding
        )
        protein_embedding = (
            self.v_embedding if protein_embedding is None else protein_embedding
        )
        dot_logits = tf.reduce_sum(
            tf.multiply(compound_embedding, protein_embedding), 1
        )
        decoder_type = self.pair_decoder['type']
        if decoder_type == 'dot':
            return dot_logits
        if decoder_type == 'bilinear':
            transformed_compound = tf.matmul(
                compound_embedding, self.weights['decoder_bilinear']
            )
            return tf.reduce_sum(transformed_compound * protein_embedding, axis=1)
        if decoder_type == 'mlp':
            pair_features = tf.concat(
                [
                    compound_embedding,
                    protein_embedding,
                    compound_embedding * protein_embedding,
                    tf.abs(compound_embedding - protein_embedding),
                ],
                axis=1,
            )
            hidden = tf.nn.leaky_relu(
                tf.matmul(pair_features, self.weights['decoder_mlp_hidden'])
                + self.weights['decoder_mlp_hidden_bias'],
                alpha=0.2,
            )
            residual = tf.reshape(
                tf.matmul(hidden, self.weights['decoder_mlp_output'])
                + self.weights['decoder_mlp_output_bias'],
                [-1],
            )
            return dot_logits + residual
        raise ValueError('Unsupported pair decoder: %s' % decoder_type)

    def buildPairLogits(
            self, compound_embedding=None, protein_embedding=None,
            include_jaccard=True):
        compound_embedding = (
            self.u_embedding if compound_embedding is None else compound_embedding
        )
        protein_embedding = (
            self.v_embedding if protein_embedding is None else protein_embedding
        )
        logits = self.buildBasePairLogits(
            compound_embedding, protein_embedding
        )
        if self.support_state_routing['enabled']:
            logits *= self.support_state_pair_gates['base']
        if self.support_experts['enabled']:
            logits *= 1.0 - self.support_expert_cold_gate
        if self.inductive_context['suppress_base_zero_support']:
            logits *= self.inductive_base_gate
        if self.context_terms['compound_disease']:
            logits += tf.reduce_sum(
                compound_embedding * self.v_context_embedding
                * self.weights['context_compound_disease'], axis=1
            )
        if self.context_terms['herb_protein']:
            herb_protein_logits = tf.reduce_sum(
                self.u_context_embedding * protein_embedding
                * self.weights['context_herb_protein'], axis=1
            )
            if self.support_router['enabled']:
                herb_protein_logits *= self.support_context_gate
            if self.support_state_routing['enabled']:
                herb_protein_logits *= self.support_state_pair_gates[
                    'herb_protein'
                ]
            if self.support_experts['enabled']:
                herb_protein_logits *= 1.0 - self.support_expert_cold_gate
            logits += herb_protein_logits
            if self.support_experts['enabled']:
                cold_features = self.u_context_embedding * protein_embedding
                if self.support_experts['detach_cold_features']:
                    cold_features = tf.stop_gradient(cold_features)
                logits += self.support_expert_cold_gate * tf.reduce_sum(
                    cold_features
                    * self.weights['context_herb_protein_cold'],
                    axis=1,
                )
            if self.herb_context_attention['mode'] == 'target_residual_attention':
                context_delta = (
                    self.target_herb_context_embedding - self.u_context_embedding
                )
                logits += tf.reduce_sum(
                    context_delta * protein_embedding
                    * self.weights['context_target_herb_residual'], axis=1
                )
        if self.context_terms['herb_disease']:
            herb_disease_logits = tf.reduce_sum(
                self.u_context_embedding * self.v_context_embedding
                * self.weights['context_herb_disease'], axis=1
            )
            if self.support_state_routing['enabled']:
                herb_disease_logits *= self.support_state_pair_gates[
                    'herb_disease'
                ]
            logits += herb_disease_logits
        if self.herb_prototype_transfer['enabled']:
            logits += (
                self.weights['herb_prototype_scale'][0]
                * self.herb_prototype_pair_residual
            )
            if include_jaccard and self.herb_jaccard_pair_residual is not None:
                logits += (
                    self.weights['herb_jaccard_scale'][0]
                    * self.herb_jaccard_pair_residual
                )
        return logits

    def buildHCRankingDistillationBatch(self, rng):
        settings = self.hc_ranking_distillation
        return build_distillation_candidate_batch(
            rng,
            self.hc_distillation_positive_pairs,
            self.hc_distillation_positive_targets,
            self.num_proteins,
            self.herb_prototypes,
            self.jaccard_transfer,
            self.compound_herb_indices,
            self.compound_herb_mask,
            query_count=settings['queries_per_update'],
            hard_negative_count=settings['hard_negatives'],
            random_negative_count=settings['random_negatives'],
            hard_pool_size=settings['hard_pool_size'],
            prior_strength=self.herb_prototype_transfer['prior_strength'],
        )

    def buildSafeListwiseBatch(self, rng):
        settings = self.safe_listwise_reranker
        return build_safe_listwise_candidate_batch(
            rng,
            self.safe_listwise_positive_targets,
            self.num_proteins,
            self.herb_prototypes,
            self.jaccard_transfer,
            self.compound_herb_indices,
            self.compound_herb_mask,
            settings['base_expert'],
            self.safe_listwise_support_features,
            query_count=settings['queries_per_update'],
            hard_unlabeled_count=settings['hard_unlabeled'],
            random_unlabeled_count=settings['random_unlabeled'],
            prior_strength=self.herb_prototype_transfer['prior_strength'],
            base_rank_cache=self.safe_listwise_base_rank_cache,
        )

    def buildSafeListwiseScores(self, base_scores, support_features):
        weights = self.safe_listwise_weights
        pair_features = tf.concat([
            self.u_context_embedding,
            self.v_embedding,
            self.u_context_embedding * self.v_embedding,
        ], axis=1)
        hidden = tf.nn.relu(
            tf.matmul(pair_features, weights['residual_hidden'])
            + weights['residual_hidden_bias']
        )
        residual = tf.reshape(
            tf.tanh(
                tf.matmul(hidden, weights['residual_output'])
                + weights['residual_output_bias']
            ),
            [-1],
        )
        gate_hidden = tf.nn.relu(
            tf.matmul(support_features, weights['gate_hidden'])
            + weights['gate_hidden_bias']
        )
        gate = tf.reshape(
            tf.sigmoid(
                tf.matmul(gate_hidden, weights['gate_output'])
                + weights['gate_output_bias']
            ),
            [-1],
        ) * tf.sigmoid(weights['gate_scale_raw'][0])
        scores = (
            base_scores
            + self.safe_listwise_reranker['tau'] * gate * residual
        )
        return scores, gate, residual

    def safeListwisePairScores(
            self, state, compound_indices, protein_indices, include_head=True):
        compound_indices = np.asarray(compound_indices, dtype=np.int64).reshape(-1)
        protein_indices = np.asarray(protein_indices, dtype=np.int64).reshape(-1)
        if compound_indices.shape != protein_indices.shape:
            raise ValueError('Safe listwise pair indices must match.')
        base = np.empty(compound_indices.size, dtype=np.float32)
        for compound in np.unique(compound_indices):
            positions = np.flatnonzero(compound_indices == compound)
            full_scores = self.safe_listwise_base_rank_cache.get(int(compound))
            if full_scores is None:
                full_scores = full_candidate_base_scores(
                    self.safe_listwise_reranker['base_expert'],
                    int(compound),
                    self.num_proteins,
                    self.herb_prototypes,
                    self.jaccard_transfer,
                    self.compound_herb_indices,
                    self.compound_herb_mask,
                    prior_strength=self.herb_prototype_transfer[
                        'prior_strength'
                    ],
                )['rank_percentile']
                self.safe_listwise_base_rank_cache[int(compound)] = full_scores
            base[positions] = full_scores[protein_indices[positions]]
        if not include_head:
            return base

        weights = state['safe_listwise_weights']
        contexts = state['compound_context'][compound_indices]
        proteins = state['protein'][protein_indices]
        features = np.concatenate(
            [contexts, proteins, contexts * proteins], axis=1
        )
        hidden = np.maximum(
            features.dot(weights['residual_hidden'])
            + weights['residual_hidden_bias'],
            0.0,
        )
        residual = np.tanh(
            hidden.dot(weights['residual_output']).reshape(-1)
            + np.asarray(weights['residual_output_bias']).reshape(-1)[0]
        )
        support = self.safe_listwise_support_features[compound_indices]
        gate_hidden = np.maximum(
            support.dot(weights['gate_hidden']) + weights['gate_hidden_bias'],
            0.0,
        )
        gate_logits = (
            gate_hidden.dot(weights['gate_output']).reshape(-1)
            + np.asarray(weights['gate_output_bias']).reshape(-1)[0]
        )
        gate = (
            1.0 / (1.0 + np.exp(-np.clip(gate_logits, -50, 50)))
        )
        raw_scale = float(np.asarray(weights['gate_scale_raw']).reshape(-1)[0])
        gate_scale = 1.0 / (1.0 + np.exp(-np.clip(raw_scale, -50, 50)))
        return (
            base + self.safe_listwise_reranker['tau']
            * gate * gate_scale * residual
        ).astype(np.float32)

    def evaluateSafeListwiseValidation(self, state, include_head=True):
        protein_ids = [
            str(self.data.id2protein[index])
            for index in range(self.num_proteins)
        ]

        def score_pairs(compound_ids, target_ids):
            compound_indices = np.asarray([
                self.data.compound[str(value)] for value in compound_ids
            ], dtype=np.int64)
            protein_indices = np.asarray([
                self.data.protein[str(value)] for value in target_ids
            ], dtype=np.int64)
            return self.safeListwisePairScores(
                state,
                compound_indices,
                protein_indices,
                include_head=include_head,
            )

        return evaluate_fixed_candidate_ranking(
            protein_ids,
            self.data.trainingData,
            self.validationData,
            score_pairs,
            ks=(20,),
            export_top=0,
        )

    def evaluateSafeListwiseSampledValidation(self, state, include_head=True):
        compound_indices = np.asarray([
            self.data.compound[str(row[0])] for row in self.validationData
        ], dtype=np.int64)
        protein_indices = np.asarray([
            self.data.protein[str(row[1])] for row in self.validationData
        ], dtype=np.int64)
        labels = np.asarray([
            1 if float(row[2]) > 0 else 0 for row in self.validationData
        ], dtype=np.int32)
        if np.unique(labels).size < 2:
            raise ValueError(
                'Safe listwise sampled validation requires both labels.'
            )
        scores = self.safeListwisePairScores(
            state,
            compound_indices,
            protein_indices,
            include_head=include_head,
        )
        return {
            'SampledAUPR': float(average_precision_score(labels, scores)),
            'SampledAUC': float(roc_auc_score(labels, scores)),
        }

    def supportContextGateValues(self, state, compound_indices):
        if not self.support_router['enabled']:
            return None
        compound_indices = np.asarray(compound_indices, dtype=np.int64)
        raw_slope = np.asarray(
            state['weights']['support_router_raw_slope']
        ).reshape(-1)
        slope = softplus(raw_slope[0])
        return monotonic_context_gate(
            self.compound_cp_support_degrees[compound_indices],
            self.compound_context_available[compound_indices],
            slope,
        )

    def supportExpertColdGateValues(self, compound_indices):
        if not self.support_experts['enabled']:
            return None
        compound_indices = np.asarray(compound_indices, dtype=np.int64)
        return support_conditioned_cold_gate(
            self.compound_cp_support_degrees[compound_indices],
            self.compound_context_available[compound_indices],
        )

    def inductiveBaseGateValues(self, compound_indices):
        if not self.inductive_context['suppress_base_zero_support']:
            return None
        compound_indices = np.asarray(compound_indices, dtype=np.int64)
        return support_decoupled_base_gate(
            self.compound_cp_support_degrees[compound_indices],
            self.compound_inductive_context_available[compound_indices],
        )

    def supportStatePairGateValues(
            self, compound_indices, protein_indices):
        if not self.support_state_routing['enabled']:
            return None
        compound_indices = np.asarray(compound_indices, dtype=np.int64)
        protein_indices = np.asarray(protein_indices, dtype=np.int64)
        return support_state_pair_gates(
            self.compound_cp_support_degrees[compound_indices],
            self.protein_cp_support_degrees[protein_indices],
            self.compound_context_available[compound_indices],
            self.protein_context_available[protein_indices],
        )

    def buildSupportStateAuxiliaryLosses(self):
        if not self.support_state_routing['enabled']:
            zero = tf.constant(0.0, dtype=tf.float32)
            return zero, zero
        herb_protein_loss = tf.constant(0.0, dtype=tf.float32)
        if (
            self.support_state_routing['training_mode']
            == 'isolated_heads'
        ):
            herb_protein_features = tf.stop_gradient(
                self.u_context_embedding * self.v_embedding
            )
            herb_protein_logits = tf.reduce_sum(
                herb_protein_features
                * self.weights['context_herb_protein'],
                axis=1,
            )
            herb_protein_loss = (
                self.support_state_routing['herb_protein_aux_weight']
                * tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=self.neg_disease_embedding,
                        logits=herb_protein_logits,
                    )
                )
            )
        cold_cold_features = (
            self.u_context_embedding * self.v_context_embedding
        )
        if self.support_state_routing['detach_cold_cold_features']:
            cold_cold_features = tf.stop_gradient(cold_cold_features)
        auxiliary_logits = tf.reduce_sum(
            cold_cold_features * self.weights['context_herb_disease'],
            axis=1,
        )
        raw_loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.neg_disease_embedding,
                logits=auxiliary_logits,
            )
        )
        cold_cold_loss = (
            self.support_state_routing['cold_cold_aux_weight'] * raw_loss
        )
        return herb_protein_loss, cold_cold_loss

    def buildSupportStateAuxiliaryLoss(self):
        herb_protein_loss, cold_cold_loss = (
            self.buildSupportStateAuxiliaryLosses()
        )
        return herb_protein_loss + cold_cold_loss

    def buildContextMaskedTrainingLoss(self):
        if not self.context_mask_training['enabled']:
            return tf.constant(0.0, dtype=tf.float32)
        side = self.context_mask_training['side']
        compound_embedding = (
            self.u_context_embedding if side in {'compound', 'both'}
            else self.u_embedding
        )
        protein_embedding = (
            self.v_context_embedding if side in {'protein', 'both'}
            else self.v_embedding
        )
        masked_logits = self.buildPairLogits(
            compound_embedding=compound_embedding,
            protein_embedding=protein_embedding,
        )
        raw_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.neg_disease_embedding,
            logits=masked_logits,
        ))
        return self.context_mask_training['weight'] * raw_loss

    def buildRegularizationLoss(self):
        weight_decay = float(self.config['weight.reg']) if self.config.contains('weight.reg') else 0.01
        return build_regularization_loss(
            tf,
            self.weights,
            self.final_uembedding,
            self.final_iembedding,
            weight_decay,
            self.regU,
            self.regI,
        )

    def buildCounterfactualContextLoss(self):
        if not self.counterfactual_context['enabled']:
            zero = tf.constant(0.0, dtype=tf.float32)
            return zero, zero, zero

        counterfactual_context_embedding = tf.nn.embedding_lookup(
            self.final_compound_context, self.counterfactual_u_idx
        )
        factual_context_logits = tf.reduce_sum(
            self.u_context_embedding * self.v_embedding
            * self.weights['context_herb_protein'], axis=1
        )
        counterfactual_context_logits = tf.reduce_sum(
            counterfactual_context_embedding * self.v_embedding
            * self.weights['context_herb_protein'], axis=1
        )
        if self.support_router['enabled']:
            factual_context_logits *= self.support_context_gate
            counterfactual_context_logits *= self.support_context_gate
        if self.support_experts['enabled']:
            warm_gate = 1.0 - self.support_expert_cold_gate
            factual_context_logits *= warm_gate
            counterfactual_context_logits *= warm_gate
            factual_cold_features = (
                self.u_context_embedding * self.v_embedding
            )
            counterfactual_cold_features = (
                counterfactual_context_embedding * self.v_embedding
            )
            if self.support_experts['detach_cold_features']:
                factual_cold_features = tf.stop_gradient(
                    factual_cold_features
                )
                counterfactual_cold_features = tf.stop_gradient(
                    counterfactual_cold_features
                )
            factual_context_logits += (
                self.support_expert_cold_gate * tf.reduce_sum(
                    factual_cold_features
                    * self.weights['context_herb_protein_cold'],
                    axis=1,
                )
            )
            counterfactual_context_logits += (
                self.support_expert_cold_gate * tf.reduce_sum(
                    counterfactual_cold_features
                    * self.weights['context_herb_protein_cold'],
                    axis=1,
                )
            )
        positive_mask = tf.cast(
            self.neg_disease_embedding > 0.5, tf.float32
        )
        active_mask = positive_mask * self.counterfactual_eligible_mask
        margins = factual_context_logits - counterfactual_context_logits
        violations = tf.nn.relu(
            self.counterfactual_context['margin'] - margins
        ) * active_mask
        raw_loss = tf.reduce_sum(violations)
        weighted_loss = self.counterfactual_context['weight'] * raw_loss
        active_count = tf.reduce_sum(active_mask)
        active_mean_margin = tf.reduce_sum(margins * active_mask) / tf.maximum(
            active_count, 1.0
        )
        return weighted_loss, active_count, active_mean_margin

    def fetchModelState(self, include_context_audit=False):
        fetches = {
            'compound': self.final_uembedding,
            'protein': self.final_iembedding,
            'compound_context': self.final_scoring_compound_context,
            'inclusive_compound_context': self.final_compound_context,
            'protein_context': self.final_protein_context,
            'herb_edge': self.final_hcedge,
            'disease_edge': self.final_pdedge,
            'weights': self.weights,
        }
        if self.safe_listwise_reranker['enabled']:
            fetches['safe_listwise_weights'] = self.safe_listwise_weights
        if include_context_audit:
            fetches['hc_edge_inputs'] = self.final_hc_edge_inputs
        return self.sess.run(fetches, feed_dict={self.isTraining: 0})

    def targetConditionedHerbContexts(
            self, state, compound_indices, protein_indices):
        if self.herb_context_attention['mode'] not in TARGET_HERB_ATTENTION_MODES:
            return None, None
        weights = state['weights']
        return target_conditioned_herb_contexts(
            state['herb_edge'],
            self.compound_herb_indices,
            self.compound_herb_mask,
            state['protein'],
            compound_indices,
            protein_indices,
            weights['target_herb_projection'],
            weights['target_protein_projection'],
            temperature=self.herb_context_attention['temperature'],
        )

    def evaluateValidation(
            self, state, metric, mask_compound=False, mask_protein=False,
            validation_records=None):
        if (
            validation_records is None
            and getattr(self, 'validationDataByState', None)
        ):
            state_values = {}
            for support_state in (
                    'warm_warm', 'cold_warm', 'warm_cold', 'cold_cold'):
                records = self.validationDataByState.get(support_state)
                if not records:
                    raise ValueError(
                        'Four-state validation is missing %s records.' %
                        support_state
                    )
                state_values[support_state] = self.evaluateValidation(
                    state,
                    metric,
                    mask_compound=mask_compound,
                    mask_protein=mask_protein,
                    validation_records=records,
                )
            macro_value = float(np.mean(list(state_values.values())))
            self.validation_state_metrics = state_values
            print(
                'validation support states: %s macro-%s=%.6f' % (
                    ' '.join(
                        '%s=%.6f' % (name, state_values[name])
                        for name in (
                            'warm_warm', 'cold_warm',
                            'warm_cold', 'cold_cold',
                        )
                    ),
                    metric.upper(),
                    macro_value,
                )
            )
            return macro_value

        records = (
            self.validationData
            if validation_records is None else validation_records
        )
        if not records:
            raise ValueError('Early stopping is enabled but no inner validation records were provided.')

        compound_indices = []
        protein_indices = []
        labels = []
        for compound_id, protein_id, rating in records:
            compound_key = str(compound_id)
            protein_key = str(protein_id)
            if compound_key not in self.data.compound or protein_key not in self.data.protein:
                raise ValueError(
                    'Validation pair contains an entity outside the model universe: %s, %s.' %
                    (compound_key, protein_key)
                )
            compound_indices.append(self.data.compound[compound_key])
            protein_indices.append(self.data.protein[protein_key])
            labels.append(1 if float(rating) > 0 else 0)

        labels = np.asarray(labels, dtype=np.int32)
        if len(np.unique(labels)) < 2:
            raise ValueError('Inner validation must contain both positive and negative records.')
        zero_weight = np.zeros(self.emb_size, dtype=np.float32)
        weights = state['weights']
        if mask_compound or mask_protein:
            from util.model_components import context_masked_pair_scores
            logits = context_masked_pair_scores(
                state['compound'],
                state['protein'],
                state['compound_context'],
                state['protein_context'],
                compound_indices,
                protein_indices,
                weights.get('context_compound_disease', zero_weight),
                weights.get('context_herb_protein', zero_weight),
                weights.get('context_herb_disease', zero_weight),
                mask_compound=mask_compound,
                mask_protein=mask_protein,
                enabled_terms=self.context_terms,
                decoder_type=self.pair_decoder['type'],
                decoder_weights=weights,
            )
        else:
            support_state_gates = self.supportStatePairGateValues(
                compound_indices, protein_indices
            )
            target_compound_contexts, _ = self.targetConditionedHerbContexts(
                state, compound_indices, protein_indices
            )
            pair_compound_contexts = (
                target_compound_contexts
                if self.herb_context_attention['mode'] == 'target_attention'
                else None
            )
            residual_compound_contexts = (
                target_compound_contexts
                if self.herb_context_attention['mode'] == 'target_residual_attention'
                else None
            )
            logits = context_interaction_pair_scores(
                state['compound'],
                state['protein'],
                state['compound_context'],
                state['protein_context'],
                compound_indices,
                protein_indices,
                weights.get('context_compound_disease', zero_weight),
                weights.get('context_herb_protein', zero_weight),
                weights.get('context_herb_disease', zero_weight),
                enabled_terms=self.context_terms,
                decoder_type=self.pair_decoder['type'],
                decoder_weights=weights,
                pair_compound_contexts=pair_compound_contexts,
                residual_compound_contexts=residual_compound_contexts,
                target_residual_weight=weights.get('context_target_herb_residual'),
                herb_protein_scale=(
                    support_state_gates['herb_protein']
                    if support_state_gates is not None
                    else (
                        1.0 - self.supportExpertColdGateValues(
                            compound_indices
                        )
                        if self.support_experts['enabled']
                        else self.supportContextGateValues(
                            state, compound_indices
                        )
                    )
                ),
                base_score_scale=(
                    support_state_gates['base']
                    if support_state_gates is not None
                    else (
                        1.0 - self.supportExpertColdGateValues(
                            compound_indices
                        )
                        if self.support_experts['enabled']
                        else self.inductiveBaseGateValues(compound_indices)
                    )
                ),
                cold_herb_protein_weight=weights.get(
                    'context_herb_protein_cold'
                ),
                cold_herb_protein_scale=self.supportExpertColdGateValues(
                    compound_indices
                ),
                herb_disease_scale=(
                    support_state_gates['herb_disease']
                    if support_state_gates is not None else None
                ),
                herb_prototype_residual=self.herbPrototypePairResiduals(
                    compound_indices, protein_indices
                ),
                herb_prototype_scale=weights.get('herb_prototype_scale'),
                herb_jaccard_residual=self.herbJaccardPairResiduals(
                    compound_indices, protein_indices
                ) if (
                    self.herb_prototype_transfer['dual_granularity']
                    and self.herb_prototype_transfer['dual_mode']
                    == 'independent_scales'
                ) else None,
                herb_jaccard_scale=weights.get('herb_jaccard_scale'),
            )
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
        if metric == 'aupr':
            return float(average_precision_score(labels, scores))
        if metric == 'auc':
            return float(roc_auc_score(labels, scores))
        raise ValueError('Unsupported validation metric: %s' % metric)

    def trainModel(self):
        def format_duration(seconds):
            seconds = max(0, int(seconds))
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            if hours:
                return '%dh%02dm%02ds' % (hours, minutes, secs)
            if minutes:
                return '%dm%02ds' % (minutes, secs)
            return '%ds' % secs

        if (
            self.support_state_routing['enabled']
            and self.support_state_routing['training_mode']
            == 'isolated_heads'
        ):
            logits = self.buildBasePairLogits()
        else:
            logits = self.buildPairLogits()
        bce_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.neg_disease_embedding,
            logits=logits
        ))

        reg_loss = self.buildRegularizationLoss()
        counterfactual_loss, counterfactual_active, counterfactual_margin = (
            self.buildCounterfactualContextLoss()
        )
        context_mask_loss = self.buildContextMaskedTrainingLoss()
        (
            support_state_hp_auxiliary_loss,
            support_state_cc_auxiliary_loss,
        ) = self.buildSupportStateAuxiliaryLosses()
        support_state_auxiliary_loss = (
            support_state_hp_auxiliary_loss
            + support_state_cc_auxiliary_loss
        )
        loss = (
            bce_loss
            + reg_loss
            + counterfactual_loss
            + context_mask_loss
            + support_state_auxiliary_loss
        )

        optimizer = tf.train.AdamOptimizer(self.lRate)
        train = optimizer.minimize(loss)

        distillation_train = None
        distillation_loss = tf.constant(0.0, dtype=tf.float32)
        distillation_raw_loss = tf.constant(0.0, dtype=tf.float32)
        if self.hc_ranking_distillation['enabled']:
            candidate_count = self.hc_ranking_distillation['candidate_count']
            self.hc_distillation_teacher_scores = tf.placeholder(
                tf.float32,
                shape=[None, candidate_count],
                name='hc_distillation_teacher_scores',
            )
            self.hc_distillation_jaccard_residuals = tf.placeholder(
                tf.float32,
                shape=[None, candidate_count],
                name='hc_distillation_jaccard_residuals',
            )
            distillation_logits = self.buildPairLogits(
                include_jaccard=False
            ) + self.weights['herb_jaccard_scale'][0] * tf.reshape(
                self.hc_distillation_jaccard_residuals, [-1]
            )
            distillation_logits = tf.reshape(
                distillation_logits, [-1, candidate_count]
            )
            temperature = tf.constant(
                self.hc_ranking_distillation['temperature'], tf.float32
            )
            teacher_log_probabilities = tf.nn.log_softmax(
                tf.stop_gradient(self.hc_distillation_teacher_scores)
                / temperature,
                axis=1,
            )
            teacher_probabilities = tf.exp(teacher_log_probabilities)
            student_log_probabilities = tf.nn.log_softmax(
                distillation_logits / temperature, axis=1
            )
            distillation_raw_loss = tf.reduce_mean(tf.reduce_sum(
                teacher_probabilities * (
                    teacher_log_probabilities - student_log_probabilities
                ),
                axis=1,
            ))
            distillation_loss = (
                self.hc_ranking_distillation['weight']
                * temperature * temperature
                * distillation_raw_loss
            )
            distillation_train = optimizer.minimize(distillation_loss)

        safe_listwise_train = None
        safe_listwise_loss = tf.constant(0.0, dtype=tf.float32)
        safe_listwise_components = {}
        if self.safe_listwise_reranker['enabled']:
            self.safe_listwise_base_scores = tf.placeholder(
                tf.float32, shape=[None], name='safe_listwise_base_scores'
            )
            self.safe_listwise_support = tf.placeholder(
                tf.float32, shape=[None, 4], name='safe_listwise_support'
            )
            self.safe_listwise_labels = tf.placeholder(
                tf.float32, shape=[None], name='safe_listwise_labels'
            )
            self.safe_listwise_valid_mask = tf.placeholder(
                tf.float32, shape=[None], name='safe_listwise_valid_mask'
            )
            self.safe_listwise_hard_mask = tf.placeholder(
                tf.float32, shape=[None], name='safe_listwise_hard_mask'
            )
            self.safe_listwise_query_width = tf.placeholder(
                tf.int32, shape=[], name='safe_listwise_query_width'
            )
            safe_scores, safe_gate, safe_residual = self.buildSafeListwiseScores(
                self.safe_listwise_base_scores,
                self.safe_listwise_support,
            )
            width = self.safe_listwise_query_width
            score_rows = tf.reshape(safe_scores, [-1, width])
            base_rows = tf.reshape(self.safe_listwise_base_scores, [-1, width])
            label_rows = tf.reshape(self.safe_listwise_labels, [-1, width])
            valid_rows = tf.reshape(self.safe_listwise_valid_mask, [-1, width])
            hard_rows = tf.reshape(self.safe_listwise_hard_mask, [-1, width])
            negative_large = tf.constant(-1e9, dtype=tf.float32)
            temperature = tf.constant(
                self.safe_listwise_reranker['temperature'], tf.float32
            )
            valid_logits = tf.where(
                valid_rows > 0,
                score_rows / temperature,
                tf.fill(tf.shape(score_rows), negative_large),
            )
            positive_logits = tf.where(
                label_rows > 0,
                score_rows / temperature,
                tf.fill(tf.shape(score_rows), negative_large),
            )
            listwise_loss = tf.reduce_mean(
                tf.reduce_logsumexp(valid_logits, axis=1)
                - tf.reduce_logsumexp(positive_logits, axis=1)
            )
            positive_count = tf.reduce_sum(label_rows, axis=1)
            positive_mean = tf.reduce_sum(
                score_rows * label_rows, axis=1
            ) / tf.maximum(positive_count, 1.0)
            pairwise_values = tf.nn.softplus(
                score_rows - tf.expand_dims(positive_mean, axis=1)
            ) * hard_rows
            pairwise_loss = tf.reduce_sum(pairwise_values) / tf.maximum(
                tf.reduce_sum(hard_rows), 1.0
            )
            bce_values = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.safe_listwise_labels,
                logits=safe_scores,
            ) * self.safe_listwise_valid_mask
            sampled_bce_loss = tf.reduce_sum(bce_values) / tf.maximum(
                tf.reduce_sum(self.safe_listwise_valid_mask), 1.0
            )
            trust_values = tf.square(
                score_rows - base_rows
            ) * valid_rows
            trust_loss = tf.reduce_sum(trust_values) / tf.maximum(
                tf.reduce_sum(valid_rows), 1.0
            )
            safe_listwise_loss = (
                self.safe_listwise_reranker['listwise_weight'] * listwise_loss
                + self.safe_listwise_reranker['pairwise_weight'] * pairwise_loss
                + self.safe_listwise_reranker['bce_weight'] * sampled_bce_loss
                + self.safe_listwise_reranker['trust_weight'] * trust_loss
            )
            safe_listwise_components = {
                'listwise': listwise_loss,
                'pairwise': pairwise_loss,
                'bce': sampled_bce_loss,
                'trust': trust_loss,
                'gate_mean': tf.reduce_mean(safe_gate),
                'residual_abs_mean': tf.reduce_mean(tf.abs(safe_residual)),
            }
            safe_listwise_optimizer = tf.train.AdamOptimizer(
                self.lRate, name='safe_listwise_adam'
            )
            safe_listwise_train = safe_listwise_optimizer.minimize(
                safe_listwise_loss,
                var_list=list(self.safe_listwise_weights.values()),
            )


        init = tf.global_variables_initializer()
        self.sess.run(init)

        early_stopping = resolve_early_stopping(self.config)
        if self.safe_listwise_reranker['enabled'] and early_stopping['enabled']:
            raise ValueError(
                'Disable sampled-pair early.stopping when safe listwise '
                'full-candidate early stopping is enabled.'
            )
        if early_stopping['enabled'] and not self.validationData:
            raise ValueError('early.stopping=True requires a non-empty inner validation split.')
        if self.safe_listwise_reranker['enabled'] and not self.validationData:
            raise ValueError(
                'Safe listwise reranking requires a non-empty inner validation split.'
            )
        tracker = None
        best_validation_state_metrics = None
        if early_stopping['enabled']:
            tracker = EarlyStoppingTracker(
                early_stopping['patience'], early_stopping['min_delta']
            )
            print(
                'Early stopping: metric=%s interval=%d patience=%d min_delta=%g validation_pairs=%d' % (
                    early_stopping['metric'].upper(),
                    early_stopping['interval'],
                    early_stopping['patience'],
                    early_stopping['min_delta'],
                    len(self.validationData),
                )
            )

        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        model_dir = os.path.join("./saved_model", currentTime)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "hdcti_model.ckpt")
        saver = tf.train.Saver(max_to_keep=1)

        safe_tracker = None
        safe_baseline_metrics = None
        safe_best_metrics = None
        if self.safe_listwise_reranker['enabled']:
            safe_tracker = EarlyStoppingTracker(
                self.safe_listwise_reranker['patience'],
                self.safe_listwise_reranker['min_delta'],
            )
            safe_baseline = self.evaluateSafeListwiseValidation(
                self.fetchModelState(), include_head=False
            )
            safe_baseline_metrics = safe_baseline['metrics']
            safe_baseline_metrics.update(
                self.evaluateSafeListwiseSampledValidation(
                    self.fetchModelState(), include_head=False
                )
            )
            safe_tracker.update(safe_baseline_metrics['MRR'], 0)
            safe_best_metrics = dict(safe_baseline_metrics)
            saver.save(self.sess, model_path)
            print(
                'Safe listwise baseline: MRR=%.6f Recall@20=%.6f '
                'sampled_AUPR=%.6f; '
                'selection uses inner validation only.' % (
                    safe_baseline_metrics['MRR'],
                    safe_baseline_metrics['Recall@20'],
                    safe_baseline_metrics['SampledAUPR'],
                )
            )

        total_batches = int(np.ceil(float(self.train_size) / self.batch_size))
        total_steps = max(1, self.maxEpoch * total_batches)
        train_start = time()
        epochs_completed = 0
        distillation_steps = 0
        distillation_last_raw_loss = None
        safe_listwise_steps = 0
        safe_listwise_last = None
        fold_digits = ''.join(
            character for character in str(self.foldInfo)
            if character.isdigit()
        )
        distillation_seed = (
            int(self.config['random.seed'])
            if self.config.contains('random.seed') else 2026
        ) + self.hc_ranking_distillation['seed_offset'] + (
            int(fold_digits) if fold_digits else 0
        ) * 1009
        distillation_rng = np.random.RandomState(distillation_seed)
        safe_listwise_seed = (
            int(self.config['random.seed'])
            if self.config.contains('random.seed') else 2026
        ) + self.safe_listwise_reranker['seed_offset'] + (
            int(fold_digits) if fold_digits else 0
        ) * 1013
        safe_listwise_rng = np.random.RandomState(safe_listwise_seed)
        training_batch_step = 0
        for epoch in range(self.maxEpoch):
            epoch_start = time()
            for n, batch in enumerate(self.next_batch_pairwise()):
                batch_start = time()
                herb_idx, i_idx, j_idx = batch
                feed_dict = {
                    self.u_idx: herb_idx,
                    self.neg_idx: j_idx,
                    self.v_idx: i_idx,
                    self.isTraining: 1,
                }
                if self.counterfactual_context['enabled']:
                    compound_batch = np.asarray(herb_idx, dtype=np.int32)
                    draw_index = epoch % self.counterfactual_context['draws']
                    feed_dict[self.counterfactual_u_idx] = (
                        self.counterfactual_donor_indices[
                            compound_batch, draw_index
                        ]
                    )
                    feed_dict[self.counterfactual_eligible_mask] = (
                        self.counterfactual_donor_eligible[compound_batch]
                    )
                (
                    _, l, cf_l, cf_active, cf_margin, cmit_l,
                    support_state_hp_aux_l,
                    support_state_cc_aux_l,
                ) = self.sess.run(
                    [
                        train,
                        loss,
                        counterfactual_loss,
                        counterfactual_active,
                        counterfactual_margin,
                        context_mask_loss,
                        support_state_hp_auxiliary_loss,
                        support_state_cc_auxiliary_loss,
                    ],
                    feed_dict=feed_dict,
                )
                if not np.isfinite(l):
                    raise ValueError('Training loss became non-finite at epoch %d batch %d: %s' % (epoch + 1, n, l))
                training_batch_step += 1
                batch_distillation_loss = None
                if (
                    distillation_train is not None
                    and training_batch_step
                    % self.hc_ranking_distillation['interval'] == 0
                ):
                    distillation_batch = self.buildHCRankingDistillationBatch(
                        distillation_rng
                    )
                    distillation_feed = {
                        self.u_idx: distillation_batch['compound_indices'],
                        self.v_idx: distillation_batch['protein_indices'],
                        self.isTraining: 1,
                        self.hc_distillation_teacher_scores: (
                            distillation_batch['teacher_scores']
                        ),
                        self.hc_distillation_jaccard_residuals: (
                            distillation_batch['jaccard_residuals']
                        ),
                    }
                    (
                        _, batch_distillation_loss,
                        distillation_last_raw_loss,
                    ) = self.sess.run(
                        [
                            distillation_train,
                            distillation_loss,
                            distillation_raw_loss,
                        ],
                        feed_dict=distillation_feed,
                    )
                    if not np.isfinite(batch_distillation_loss):
                        raise ValueError(
                            'H-C ranking distillation loss became non-finite '
                            'at epoch %d batch %d: %s' % (
                                epoch + 1, n, batch_distillation_loss
                            )
                        )
                    distillation_steps += 1
                batch_safe_listwise_loss = None
                if (
                    safe_listwise_train is not None
                    and training_batch_step
                    % self.safe_listwise_reranker['interval'] == 0
                ):
                    safe_batch = self.buildSafeListwiseBatch(
                        safe_listwise_rng
                    )
                    candidate_width = safe_batch['candidate_width']
                    repeated_compounds = np.repeat(
                        safe_batch['compound_indices'], candidate_width
                    ).astype(np.int32)
                    repeated_support = np.repeat(
                        safe_batch['support_features'], candidate_width, axis=0
                    ).astype(np.float32)
                    safe_feed = {
                        self.u_idx: repeated_compounds,
                        self.v_idx: safe_batch['protein_indices'].reshape(-1),
                        self.isTraining: 1,
                        self.safe_listwise_base_scores: (
                            safe_batch['base_rank_scores'].reshape(-1)
                        ),
                        self.safe_listwise_support: repeated_support,
                        self.safe_listwise_labels: (
                            safe_batch['labels'].reshape(-1)
                        ),
                        self.safe_listwise_valid_mask: (
                            safe_batch['valid_mask'].reshape(-1)
                        ),
                        self.safe_listwise_hard_mask: (
                            safe_batch['hard_unlabeled_mask'].reshape(-1)
                        ),
                        self.safe_listwise_query_width: candidate_width,
                    }
                    component_names = tuple(safe_listwise_components)
                    outputs = self.sess.run(
                        [safe_listwise_train, safe_listwise_loss]
                        + [safe_listwise_components[name]
                           for name in component_names],
                        feed_dict=safe_feed,
                    )
                    batch_safe_listwise_loss = float(outputs[1])
                    safe_listwise_last = {
                        name: float(value)
                        for name, value in zip(component_names, outputs[2:])
                    }
                    safe_listwise_last['loss'] = batch_safe_listwise_loss
                    if not np.isfinite(batch_safe_listwise_loss):
                        raise ValueError(
                            'Safe listwise loss became non-finite at epoch '
                            '%d batch %d: %s' % (
                                epoch + 1, n, batch_safe_listwise_loss
                            )
                        )
                    safe_listwise_steps += 1
                elapsed = time() - train_start
                batch_time = time() - batch_start
                step = min(epoch * total_batches + n + 1, total_steps)
                avg_step_time = elapsed / step
                eta = avg_step_time * (total_steps - step)
                chcr_suffix = ''
                if self.counterfactual_context['enabled']:
                    chcr_suffix = (
                        ' chcr_loss: %s chcr_active: %d chcr_margin: %s '
                        'chcr_draw: %d/%d' % (
                            cf_l,
                            int(cf_active),
                            cf_margin,
                            epoch % self.counterfactual_context['draws'] + 1,
                            self.counterfactual_context['draws'],
                        )
                    )
                cmit_suffix = ''
                if self.context_mask_training['enabled']:
                    cmit_suffix = ' cmit_loss: %s' % cmit_l
                support_state_suffix = ''
                if self.support_state_routing['enabled']:
                    support_state_suffix = (
                        ' support_state_hp_aux_loss: %s'
                        ' support_state_cc_aux_loss: %s' % (
                            support_state_hp_aux_l,
                            support_state_cc_aux_l,
                        )
                    )
                distillation_suffix = ''
                if batch_distillation_loss is not None:
                    distillation_suffix = (
                        ' hcrd_loss: %s hcrd_step: %d' % (
                            batch_distillation_loss, distillation_steps
                        )
                    )
                safe_listwise_suffix = ''
                if batch_safe_listwise_loss is not None:
                    safe_listwise_suffix = (
                        ' slr_loss: %.6f slr_gate: %.6f slr_step: %d' % (
                            batch_safe_listwise_loss,
                            safe_listwise_last['gate_mean'],
                            safe_listwise_steps,
                        )
                    )
                print(
                    'training: %d/%d batch %d/%d loss: %s batch_time: %s '
                    'elapsed: %s eta: %s%s%s%s%s%s' % (
                        epoch + 1,
                        self.maxEpoch,
                        n + 1,
                        total_batches,
                        l,
                        format_duration(batch_time),
                        format_duration(elapsed),
                        format_duration(eta),
                        chcr_suffix,
                        cmit_suffix,
                        support_state_suffix,
                        distillation_suffix,
                        safe_listwise_suffix,
                    )
                )
            print('epoch %d/%d finished in %s' %
                  (epoch + 1, self.maxEpoch, format_duration(time() - epoch_start)))
            epochs_completed = epoch + 1

            safe_should_validate = (
                self.safe_listwise_reranker['enabled']
                and (
                    (epoch + 1)
                    % self.safe_listwise_reranker['validation_interval'] == 0
                    or epoch + 1 == self.maxEpoch
                )
            )
            if safe_should_validate:
                safe_ranking = self.evaluateSafeListwiseValidation(
                    self.fetchModelState(), include_head=True
                )
                safe_metrics = safe_ranking['metrics']
                safe_metrics.update(
                    self.evaluateSafeListwiseSampledValidation(
                        self.fetchModelState(), include_head=True
                    )
                )
                recall_floor = (
                    safe_baseline_metrics['Recall@20']
                    - self.safe_listwise_reranker[
                        'recall_at_20_tolerance'
                    ]
                )
                recall_allowed = safe_metrics['Recall@20'] >= recall_floor
                tracker_value = (
                    safe_metrics['MRR'] if recall_allowed else -1.0
                )
                improved, safe_should_stop = safe_tracker.update(
                    tracker_value, epoch + 1
                )
                if improved:
                    safe_best_metrics = dict(safe_metrics)
                    saver.save(self.sess, model_path)
                print(
                    'safe validation: epoch %d MRR=%.6f best=%.6f '
                    'Recall@20=%.6f sampled_AUPR=%.6f floor=%.6f '
                    'stale=%d/%d%s%s' % (
                        epoch + 1,
                        safe_metrics['MRR'],
                        safe_tracker.best_value,
                        safe_metrics['Recall@20'],
                        safe_metrics['SampledAUPR'],
                        recall_floor,
                        safe_tracker.stale_checks,
                        safe_tracker.patience,
                        ' improved' if improved else '',
                        '' if recall_allowed else ' recall_constraint_failed',
                    )
                )
                if safe_should_stop:
                    print(
                        'Safe listwise early stopping triggered at epoch %d.' %
                        (epoch + 1)
                    )
                    break

            should_validate = early_stopping['enabled'] and (
                (epoch + 1) % early_stopping['interval'] == 0
                or epoch + 1 == self.maxEpoch
            )
            if should_validate:
                validation_value = self.evaluateValidation(
                    self.fetchModelState(), early_stopping['metric']
                )
                improved, should_stop = tracker.update(validation_value, epoch + 1)
                if improved:
                    saver.save(self.sess, model_path)
                    if getattr(self, 'validation_state_metrics', None):
                        best_validation_state_metrics = dict(
                            self.validation_state_metrics
                        )
                print(
                    'validation: epoch %d %s=%.6f best=%.6f best_epoch=%d stale=%d/%d%s' % (
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
                    print('Early stopping triggered at epoch %d.' % (epoch + 1))
                    break

        if self.safe_listwise_reranker['enabled']:
            saver.restore(self.sess, model_path)
            self.early_stopping_summary = {
                'best_epoch': safe_tracker.best_epoch,
                'best_value': safe_tracker.best_value,
                'epochs_completed': epochs_completed,
                'metric': 'full_candidate_mrr',
                'aggregation': 'macro_over_compounds',
                'baseline_metrics': safe_baseline_metrics,
                'best_metrics': safe_best_metrics,
            }
            print(
                'Restored safe listwise checkpoint: epoch %d MRR=%.6f '
                'Recall@20=%.6f' % (
                    safe_tracker.best_epoch,
                    safe_tracker.best_value,
                    safe_best_metrics['Recall@20'],
                )
            )
        elif early_stopping['enabled']:
            saver.restore(self.sess, model_path)
            self.early_stopping_summary = {
                'best_epoch': tracker.best_epoch,
                'best_value': tracker.best_value,
                'epochs_completed': epochs_completed,
                'metric': early_stopping['metric'],
                'aggregation': self.validationAggregation,
            }
            if best_validation_state_metrics is not None:
                self.early_stopping_summary['state_values'] = (
                    best_validation_state_metrics
                )
            print(
                'Restored best validation checkpoint: epoch %d %s=%.6f' % (
                    tracker.best_epoch,
                    early_stopping['metric'].upper(),
                    tracker.best_value,
                )
            )
        else:
            self.early_stopping_summary = None

        state = self.fetchModelState()
        self.context_mask_summary = None
        if self.context_mask_training['enabled'] and self.validationData:
            side = self.context_mask_training['side']
            masked_value = self.evaluateValidation(
                state,
                early_stopping['metric'] if early_stopping['enabled'] else 'aupr',
                mask_compound=side in {'compound', 'both'},
                mask_protein=side in {'protein', 'both'},
            )
            self.context_mask_summary = {
                'side': side,
                'metric': (
                    early_stopping['metric'] if early_stopping['enabled'] else 'aupr'
                ),
                'value': masked_value,
            }
            print('CMIT masked validation: side=%s %s=%.6f' % (
                side,
                self.context_mask_summary['metric'].upper(),
                masked_value,
            ))
        self.u = state['compound']
        self.i = state['protein']
        self.u_context = state['compound_context']
        self.i_context = state['protein_context']
        self.herb_edge = state['herb_edge']
        self.weight = state['weights']
        self.safe_listwise_weight = state.get('safe_listwise_weights')
        self.hc_ranking_distillation_summary = None
        if self.hc_ranking_distillation['enabled']:
            self.hc_ranking_distillation_summary = dict(
                self.hc_ranking_distillation
            )
            self.hc_ranking_distillation_summary.update({
                'teacher': 'equal_rank_percentile_prototype_jaccard_loco',
                'seed': int(distillation_seed),
                'optimizer_steps': int(distillation_steps),
                'last_raw_kl': (
                    float(distillation_last_raw_loss)
                    if distillation_last_raw_loss is not None else None
                ),
                'outer_test_used_for_selection': False,
            })
            distillation_path = os.path.join(
                model_dir, 'hc_ranking_distillation.json'
            )
            with open(distillation_path, 'w', encoding='utf-8') as handle:
                json.dump(
                    self.hc_ranking_distillation_summary,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write('\n')
            print(
                'H-C ranking distillation: steps=%d last_raw_kl=%s.' % (
                    distillation_steps,
                    'N/A' if distillation_last_raw_loss is None
                    else '%.6f' % distillation_last_raw_loss,
                )
            )
            print('H-C ranking distillation metadata: %s' % distillation_path)
        self.safe_listwise_summary = None
        if self.safe_listwise_reranker['enabled']:
            self.safe_listwise_summary = dict(self.safe_listwise_reranker)
            self.safe_listwise_summary.update({
                'protocol': 'support_anchored_safe_listwise_reranker_v1',
                'feature_names': list(self.safe_listwise_feature_names),
                'seed': int(safe_listwise_seed),
                'optimizer_steps': int(safe_listwise_steps),
                'last_losses': safe_listwise_last,
                'baseline_validation': safe_baseline_metrics,
                'best_validation': safe_best_metrics,
                'best_epoch': safe_tracker.best_epoch,
                'outer_test_used_for_selection': False,
                'encoder_updated_by_listwise_optimizer': False,
            })
            safe_listwise_path = os.path.join(
                model_dir, 'safe_listwise_reranker.json'
            )
            with open(safe_listwise_path, 'w', encoding='utf-8') as handle:
                json.dump(
                    self.safe_listwise_summary,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write('\n')
            print(
                'Safe listwise reranker: steps=%d best_inner_MRR=%.6f '
                'best_inner_Recall@20=%.6f.' % (
                    safe_listwise_steps,
                    safe_best_metrics['MRR'],
                    safe_best_metrics['Recall@20'],
                )
            )
            print('Safe listwise metadata: %s' % safe_listwise_path)
        self.herb_prototype_summary = None
        if self.herb_prototype_transfer['enabled']:
            validation_compounds = np.asarray([
                self.data.compound[str(record[0])]
                for record in self.validationData
            ], dtype=np.int64)
            validation_proteins = np.asarray([
                self.data.protein[str(record[1])]
                for record in self.validationData
            ], dtype=np.int64)
            validation_labels = np.asarray([
                float(record[2]) > 0 for record in self.validationData
            ], dtype=bool)
            prototype_values, prototype_diagnostics = (
                self.herbPrototypePairResiduals(
                    validation_compounds,
                    validation_proteins,
                    return_diagnostics=True,
                )
            )
            learned_scale = float(np.asarray(
                self.weight['herb_prototype_scale']
            ).reshape(-1)[0])
            learned_jaccard_scale = None
            jaccard_diagnostics = None
            if (
                self.herb_prototype_transfer['dual_granularity']
                and self.herb_prototype_transfer['dual_mode']
                == 'independent_scales'
            ):
                _, jaccard_diagnostics = self.herbJaccardPairResiduals(
                    validation_compounds,
                    validation_proteins,
                    return_diagnostics=True,
                )
                learned_jaccard_scale = float(np.asarray(
                    self.weight['herb_jaccard_scale']
                ).reshape(-1)[0])
            self.herb_prototype_summary = {
                'mode': self.herb_prototype_transfer['mode'],
                'prior_strength': self.herb_prototype_transfer[
                    'prior_strength'
                ],
                'replace_compound_pagerank': self.herb_prototype_transfer[
                    'replace_compound_pagerank'
                ],
                'protein_pagerank_retained': self.encoder_profile[
                    'use_pagerank'
                ],
                'prototype_component_enabled': self.herb_prototype_transfer[
                    'prototype_component'
                ],
                'dual_granularity': self.herb_prototype_transfer[
                    'dual_granularity'
                ],
                'dual_mode': self.herb_prototype_transfer['dual_mode'],
                'jaccard_blend': self.herb_prototype_transfer[
                    'jaccard_blend'
                ],
                'learned_scale': learned_scale,
                'learned_jaccard_scale': learned_jaccard_scale,
                'training_positive_edges': self.herb_prototypes[
                    'training_positive_edges'
                ],
                'supported_compound_count': self.herb_prototypes[
                    'supported_compound_count'
                ],
                'validation': prototype_diagnostics,
                'jaccard_validation': jaccard_diagnostics,
                'positive_mean_residual': (
                    float(np.mean(prototype_values[validation_labels]))
                    if np.any(validation_labels) else None
                ),
                'negative_mean_residual': (
                    float(np.mean(prototype_values[~validation_labels]))
                    if np.any(~validation_labels) else None
                ),
            }
            prototype_path = os.path.join(
                model_dir, 'herb_prototype_transfer.json'
            )
            with open(prototype_path, 'w', encoding='utf-8') as handle:
                json.dump(
                    self.herb_prototype_summary,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write('\n')
            print(
                'Herb prototype transfer: scale=%.6f validation_coverage=%.4f '
                'mean_abs_residual=%.6f.' % (
                    learned_scale,
                    prototype_diagnostics['evidence_coverage'],
                    prototype_diagnostics['mean_abs_residual'],
                )
            )
            if learned_jaccard_scale is not None:
                print(
                    'H-C Jaccard transfer: scale=%.6f '
                    'validation_coverage=%.4f mean_abs_residual=%.6f.' % (
                        learned_jaccard_scale,
                        jaccard_diagnostics['evidence_coverage'],
                        jaccard_diagnostics['mean_abs_residual'],
                    )
                )
            print('Herb prototype metadata: %s' % prototype_path)
        self.global_token_attention_summary = None
        if self.global_token_attention['enabled']:
            layer_summaries = {}
            for side, tensor_rows in self.global_token_diagnostic_tensors.items():
                layer_summaries[side] = []
                for layer_index, tensors in enumerate(tensor_rows, start=1):
                    values = self.sess.run(
                        tensors, feed_dict={self.isTraining: 0}
                    )
                    row = summarize_global_token_attention(
                        values['edge_assignments'],
                        values['node_attention'],
                        values['token_embeddings'],
                    )
                    raw_gamma = float(np.asarray(self.weight[
                        '%s_global_token_gamma_%d' % (side, layer_index)
                    ]).reshape(-1)[0])
                    row.update({
                        'layer': layer_index,
                        'raw_residual_scale': raw_gamma,
                        'residual_scale': float(np.tanh(raw_gamma)),
                    })
                    layer_summaries[side].append(row)
            structure = {}
            for side, node_key, edge_key in (
                    ('hc', 'nodes', 'hyperedges'),
                    ('pd', 'nodes', 'hyperedges')):
                if not self.global_token_attention[side + '_enabled']:
                    continue
                side_structure = self.hyperedge_attention_structure[side]
                structure[side] = dict(side_structure)
                structure[side].update(global_token_attention_complexity(
                    side_structure[node_key],
                    side_structure[edge_key],
                    self.global_token_attention['tokens'],
                    self.global_token_attention['heads'],
                ))
            self.global_token_attention_summary = {
                'mode': self.global_token_attention['mode'],
                'hc_enabled': self.global_token_attention['hc_enabled'],
                'pd_enabled': self.global_token_attention['pd_enabled'],
                'tokens': self.global_token_attention['tokens'],
                'heads': self.global_token_attention['heads'],
                'temperature': self.global_token_attention['temperature'],
                'structure': structure,
                'layers': layer_summaries,
            }
            print(
                'HILGA residual scales: %s' % ', '.join(
                    '%s-L%d=%.6f' % (
                        side.upper(), row['layer'], row['residual_scale']
                    )
                    for side, rows in layer_summaries.items()
                    for row in rows
                )
            )
            global_attention_path = os.path.join(
                model_dir, 'global_token_attention.json'
            )
            with open(global_attention_path, 'w', encoding='utf-8') as handle:
                json.dump(
                    self.global_token_attention_summary,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write('\n')
            print('HILGA metadata: %s' % global_attention_path)
        self.hplga_summary = None
        if self.hplga['enabled']:
            layer_summaries = {}
            structure = {}
            for side, node_key in (('hc', 'nodes'), ('pd', 'nodes')):
                if not self.hplga[side + '_enabled']:
                    continue
                layer_summaries[side] = []
                for layer in range(1, self.n_layer + 1):
                    raw_gamma = float(np.asarray(self.weight[
                        '%s_hplga_gamma_%d' % (side, layer)
                    ]).reshape(-1)[0])
                    layer_summaries[side].append({
                        'layer': layer,
                        'raw_residual_scale': raw_gamma,
                        'residual_scale': float(np.tanh(raw_gamma)),
                    })
                side_structure = dict(
                    self.hyperedge_attention_structure[side]
                )
                side_structure.update(hplga_complexity(
                    side_structure[node_key],
                    self.emb_size,
                    self.hplga['heads'],
                ))
                structure[side] = side_structure
            self.hplga_summary = {
                'mode': self.hplga['mode'],
                'kernel': self.hplga['kernel'],
                'hc_enabled': self.hplga['hc_enabled'],
                'pd_enabled': self.hplga['pd_enabled'],
                'heads': self.hplga['heads'],
                'epsilon': self.hplga['epsilon'],
                'pagerank': {
                    'alpha': self.hplga['pagerank_alpha'],
                    'max_iter': self.hplga['pagerank_max_iter'],
                    'tol': self.hplga['pagerank_tol'],
                    'diagnostics': self.hplga_pagerank_diagnostics,
                },
                'structure': structure,
                'layers': layer_summaries,
            }
            print(
                'HPLGA residual scales: %s' % ', '.join(
                    '%s-L%d=%.6f' % (
                        side.upper(), row['layer'], row['residual_scale']
                    )
                    for side, rows in layer_summaries.items()
                    for row in rows
                )
            )
            hplga_path = os.path.join(model_dir, 'hplga.json')
            with open(hplga_path, 'w', encoding='utf-8') as handle:
                json.dump(
                    self.hplga_summary,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write('\n')
            print('HPLGA metadata: %s' % hplga_path)
        self.hyperedge_attention_summary = None
        if self.hyperedge_attention['enabled']:
            parameter_summary = {}
            for side in ('hc', 'pd'):
                if not self.hyperedge_attention[side + '_enabled']:
                    continue
                parameter_summary[side] = []
                for layer in range(1, self.n_layer + 1):
                    node_weight = self.weight[
                        '%s_hyper_node_%d' % (side, layer)
                    ]
                    edge_weight = self.weight[
                        '%s_hyper_edge_%d' % (side, layer)
                    ]
                    parameter_summary[side].append({
                        'layer': layer,
                        'node_score_mean_abs': float(
                            np.mean(np.abs(node_weight))
                        ),
                        'edge_score_mean_abs': float(
                            np.mean(np.abs(edge_weight))
                        ),
                    })
            self.hyperedge_attention_summary = {
                'mode': self.hyperedge_attention['mode'],
                'hc_enabled': self.hyperedge_attention['hc_enabled'],
                'pd_enabled': self.hyperedge_attention['pd_enabled'],
                'temperature': self.hyperedge_attention['temperature'],
                'prior_scale': self.hyperedge_attention['prior_scale'],
                'structure': self.hyperedge_attention_structure,
                'parameters': parameter_summary,
            }
            print(
                'Hyperedge attention learned mean abs: %s' % ', '.join(
                    '%s-L%d node=%.6f edge=%.6f' % (
                        side.upper(),
                        row['layer'],
                        row['node_score_mean_abs'],
                        row['edge_score_mean_abs'],
                    )
                    for side, rows in parameter_summary.items()
                    for row in rows
                )
            )
            attention_path = os.path.join(
                model_dir, 'hyperedge_attention.json'
            )
            with open(attention_path, 'w', encoding='utf-8') as handle:
                json.dump(
                    self.hyperedge_attention_summary,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write('\n')
            print('Hyperedge attention metadata: %s' % attention_path)
        self.support_router_summary = None
        if self.support_router['enabled']:
            support_slope = softplus(
                np.asarray(
                    self.weight['support_router_raw_slope']
                ).reshape(-1)[0]
            )
            all_support_gates = self.supportContextGateValues(
                state, np.arange(self.num_compounds, dtype=np.int64)
            )
            seen_mask = self.compound_cp_support_degrees > 0
            zero_mask = (
                (self.compound_cp_support_degrees == 0)
                & (self.compound_context_available > 0)
            )
            self.support_router_summary = {
                'mode': self.support_router['mode'],
                'learned_slope': float(support_slope),
                'pseudo_cold': self.data.pseudo_cold_info,
                'zero_support_compounds': int(np.sum(zero_mask)),
                'zero_support_gate_mean': (
                    float(np.mean(all_support_gates[zero_mask]))
                    if np.any(zero_mask) else None
                ),
                'seen_compounds': int(np.sum(seen_mask)),
                'seen_gate_mean': (
                    float(np.mean(all_support_gates[seen_mask]))
                    if np.any(seen_mask) else None
                ),
                'gate_min': float(np.min(all_support_gates)),
                'gate_max': float(np.max(all_support_gates)),
            }
            print(
                'Support router learned slope: %.6f; zero-support gate: %s; '
                'seen gate mean: %s.' % (
                    self.support_router_summary['learned_slope'],
                    '%.6f' % self.support_router_summary[
                        'zero_support_gate_mean'
                    ] if self.support_router_summary[
                        'zero_support_gate_mean'
                    ] is not None else 'n/a',
                    '%.6f' % self.support_router_summary['seen_gate_mean']
                    if self.support_router_summary['seen_gate_mean'] is not None
                    else 'n/a',
                )
            )
            router_path = os.path.join(model_dir, 'support_router.json')
            with open(router_path, 'w', encoding='utf-8') as handle:
                json.dump(
                    self.support_router_summary,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write('\n')
            print('Support router metadata: %s' % router_path)
        self.support_experts_summary = None
        if self.support_experts['enabled']:
            all_compound_indices = np.arange(
                self.num_compounds, dtype=np.int64
            )
            cold_gates = self.supportExpertColdGateValues(
                all_compound_indices
            )
            cold_mask = cold_gates > 0
            warm_mask = ~cold_mask
            self.support_experts_summary = {
                'mode': self.support_experts['mode'],
                'detach_cold_features': self.support_experts[
                    'detach_cold_features'
                ],
                'pseudo_cold': self.data.pseudo_cold_info,
                'cold_compounds': int(np.sum(cold_mask)),
                'warm_compounds': int(np.sum(warm_mask)),
                'context_available_compounds': int(np.sum(
                    self.compound_context_available > 0
                )),
                'warm_weight_mean_abs': float(np.mean(np.abs(
                    self.weight['context_herb_protein']
                ))),
                'cold_weight_mean_abs': float(np.mean(np.abs(
                    self.weight['context_herb_protein_cold']
                ))),
            }
            print(
                'SCHE routing: warm=%d cold=%d; Hctx-P weight mean abs '
                'warm=%.6f cold=%.6f.' % (
                    self.support_experts_summary['warm_compounds'],
                    self.support_experts_summary['cold_compounds'],
                    self.support_experts_summary[
                        'warm_weight_mean_abs'
                    ],
                    self.support_experts_summary[
                        'cold_weight_mean_abs'
                    ],
                )
            )
            experts_path = os.path.join(
                model_dir, 'support_conditioned_experts.json'
            )
            with open(experts_path, 'w', encoding='utf-8') as handle:
                json.dump(
                    self.support_experts_summary,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write('\n')
            print('SCHE metadata: %s' % experts_path)
        self.support_state_routing_summary = None
        if self.support_state_routing['enabled']:
            training_compounds = np.asarray([
                self.data.compound[str(record[0])]
                for record in self.data.trainingData
            ], dtype=np.int64)
            training_proteins = np.asarray([
                self.data.protein[str(record[1])]
                for record in self.data.trainingData
            ], dtype=np.int64)
            training_gates = self.supportStatePairGateValues(
                training_compounds, training_proteins
            )
            state_names = ('warm_warm', 'cold_warm', 'warm_cold', 'cold_cold')
            training_state_counts = {
                name: int(np.sum(training_gates['state'] == code))
                for code, name in enumerate(state_names)
            }
            self.support_state_routing_summary = {
                'mode': self.support_state_routing['mode'],
                'training_mode': self.support_state_routing[
                    'training_mode'
                ],
                'routes': {
                    'warm_warm': ['base', 'herb_protein'],
                    'cold_warm': ['herb_protein'],
                    'warm_cold': ['base'],
                    'cold_cold': ['herb_disease'],
                },
                'cold_cold_aux_weight': self.support_state_routing[
                    'cold_cold_aux_weight'
                ],
                'herb_protein_aux_weight': self.support_state_routing[
                    'herb_protein_aux_weight'
                ],
                'detach_cold_cold_features': self.support_state_routing[
                    'detach_cold_cold_features'
                ],
                'training_state_counts': training_state_counts,
                'validation_state_counts': {
                    name: len(self.validationDataByState.get(name, []))
                    for name in state_names
                },
                'herb_protein_weight_mean_abs': float(np.mean(np.abs(
                    self.weight['context_herb_protein']
                ))),
                'herb_disease_weight_mean_abs': float(np.mean(np.abs(
                    self.weight['context_herb_disease']
                ))),
            }
            print(
                'Support-state routing learned weights: Hctx-P=%.6f, '
                'Hctx-Dctx=%.6f; training states=%s.' % (
                    self.support_state_routing_summary[
                        'herb_protein_weight_mean_abs'
                    ],
                    self.support_state_routing_summary[
                        'herb_disease_weight_mean_abs'
                    ],
                    ', '.join(
                        '%s:%d' % (name, training_state_counts[name])
                        for name in state_names
                    ),
                )
            )
            support_state_path = os.path.join(
                model_dir, 'support_state_routing.json'
            )
            with open(
                    support_state_path, 'w', encoding='utf-8') as handle:
                json.dump(
                    self.support_state_routing_summary,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write('\n')
            print('Support-state routing metadata: %s' % support_state_path)
        if self.use_context_interaction:
            context_weight_values = {
                'compound_disease': np.mean(np.abs(self.weight['context_compound_disease'])),
                'herb_protein': np.mean(np.abs(self.weight['context_herb_protein'])),
                'herb_disease': np.mean(np.abs(self.weight['context_herb_disease'])),
            }
            print('Context weight mean abs: C-Dctx %s, Hctx-P %s, Hctx-Dctx %s' % (
                '%.6f' % context_weight_values['compound_disease']
                if self.context_terms['compound_disease'] else 'off',
                '%.6f' % context_weight_values['herb_protein']
                if self.context_terms['herb_protein'] else 'off',
                '%.6f' % context_weight_values['herb_disease']
                if self.context_terms['herb_disease'] else 'off',
            ))
        if self.herb_context_attention['mode'] == 'target_residual_attention':
            print(
                'Target herb residual weight mean abs: %.6f' %
                np.mean(np.abs(self.weight['context_target_herb_residual']))
            )
        self.target_attention_summary = None
        if (
            self.herb_context_attention['mode'] in TARGET_HERB_ATTENTION_MODES
            and self.validationData
        ):
            validation_compounds = [
                self.data.compound[str(row[0])] for row in self.validationData
            ]
            validation_proteins = [
                self.data.protein[str(row[1])] for row in self.validationData
            ]
            validation_target_contexts, validation_attention = self.targetConditionedHerbContexts(
                state, validation_compounds, validation_proteins
            )
            entropy = -np.sum(
                validation_attention
                * np.log(np.clip(validation_attention, 1e-12, 1.0)),
                axis=1,
            )
            self.target_attention_summary = {
                'validation_pairs': len(validation_compounds),
                'expanded_incidences': int(np.sum(
                    self.compound_herb_mask[validation_compounds]
                )),
                'mean_entropy': float(np.mean(entropy)),
                'mean_max_weight': float(np.mean(np.max(validation_attention, axis=1))),
            }
            if self.herb_context_attention['mode'] == 'target_residual_attention':
                validation_static_contexts = state['compound_context'][
                    validation_compounds
                ]
                validation_protein_embeddings = state['protein'][
                    validation_proteins
                ]
                context_delta = (
                    validation_target_contexts - validation_static_contexts
                )
                residual_logits = np.sum(
                    context_delta
                    * validation_protein_embeddings
                    * self.weight['context_target_herb_residual'],
                    axis=1,
                )
                self.target_attention_summary.update({
                    'mean_delta_norm': float(np.mean(
                        np.linalg.norm(context_delta, axis=1)
                    )),
                    'mean_abs_residual_logit': float(np.mean(
                        np.abs(residual_logits)
                    )),
                })
            print(
                'Target herb attention: validation_pairs=%d incidences=%d '
                'mean_entropy=%.6f mean_max_weight=%.6f' % (
                    self.target_attention_summary['validation_pairs'],
                    self.target_attention_summary['expanded_incidences'],
                    self.target_attention_summary['mean_entropy'],
                    self.target_attention_summary['mean_max_weight'],
                )
            )
            if self.herb_context_attention['mode'] == 'target_residual_attention':
                print(
                    'Target herb residual: mean_delta_norm=%.6f '
                    'mean_abs_logit=%.6f' % (
                        self.target_attention_summary['mean_delta_norm'],
                        self.target_attention_summary['mean_abs_residual_logit'],
                    )
                )
        os.makedirs('./results', exist_ok=True)
        if self.target_attention_summary is not None:
            attention_path = (
                './results/%s_top_herbs%s.tsv' % (
                    self.herb_context_attention['mode'], currentTime
                )
            )
            with open(attention_path, 'w', encoding='utf-8') as handle:
                handle.write(
                    'compound_id\tprotein_id\tlabel\trank\therb_id\tattention\n'
                )
                for pair_index, row in enumerate(self.validationData):
                    compound_index = validation_compounds[pair_index]
                    active_count = int(np.sum(
                        self.compound_herb_mask[compound_index]
                    ))
                    active_weights = validation_attention[pair_index, :active_count]
                    top_positions = np.argsort(active_weights)[::-1][:3]
                    for rank, position in enumerate(top_positions, start=1):
                        herb_index = int(
                            self.compound_herb_indices[compound_index, position]
                        )
                        handle.write('%s\t%s\t%s\t%d\t%s\t%.8f\n' % (
                            row[0], row[1], row[2], rank,
                            self.data.id2herb[herb_index],
                            active_weights[position],
                        ))
            print('Target herb attention details: %s' % attention_path)
        np.savetxt('./results/herbedgeDHCN_herb_embedding' + currentTime + '.txt', self.u)
        np.savetxt('./results/diseaseedgeDHCN_disease_embedding' + currentTime + '.txt', self.i)
        if self.use_context_interaction:
            np.savetxt('./results/compound_herb_context' + currentTime + '.txt', self.u_context)
            np.savetxt('./results/protein_disease_context' + currentTime + '.txt', self.i_context)
        save_path = (
            model_path
            if early_stopping['enabled'] or self.safe_listwise_reranker['enabled']
            else saver.save(self.sess, model_path)
        )
        print("模型权重保存成功: %s" % save_path)
    def predictForPairs(self, compound_indices, protein_indices):
        zero_weight = np.zeros(self.emb_size, dtype=np.float32)
        state = {
            'compound': self.u,
            'protein': self.i,
            'compound_context': self.u_context,
            'protein_context': self.i_context,
            'herb_edge': self.herb_edge,
            'weights': self.weight,
        }
        if self.safe_listwise_reranker['enabled']:
            state['safe_listwise_weights'] = self.safe_listwise_weight
            return self.safeListwisePairScores(
                state, compound_indices, protein_indices, include_head=True
            )
        target_compound_contexts, _ = self.targetConditionedHerbContexts(
            state, compound_indices, protein_indices
        )
        pair_compound_contexts = (
            target_compound_contexts
            if self.herb_context_attention['mode'] == 'target_attention'
            else None
        )
        residual_compound_contexts = (
            target_compound_contexts
            if self.herb_context_attention['mode'] == 'target_residual_attention'
            else None
        )
        support_state_gates = self.supportStatePairGateValues(
            compound_indices, protein_indices
        )
        return context_interaction_pair_scores(
            self.u,
            self.i,
            self.u_context,
            self.i_context,
            compound_indices,
            protein_indices,
            self.weight.get('context_compound_disease', zero_weight),
            self.weight.get('context_herb_protein', zero_weight),
            self.weight.get('context_herb_disease', zero_weight),
            enabled_terms=self.context_terms,
            decoder_type=self.pair_decoder['type'],
            decoder_weights=self.weight,
            pair_compound_contexts=pair_compound_contexts,
            residual_compound_contexts=residual_compound_contexts,
            target_residual_weight=self.weight.get('context_target_herb_residual'),
            herb_protein_scale=(
                support_state_gates['herb_protein']
                if support_state_gates is not None
                else (
                    1.0 - self.supportExpertColdGateValues(
                        compound_indices
                    )
                    if self.support_experts['enabled']
                    else self.supportContextGateValues(
                        state, compound_indices
                    )
                )
            ),
            base_score_scale=(
                support_state_gates['base']
                if support_state_gates is not None
                else (
                    1.0 - self.supportExpertColdGateValues(
                        compound_indices
                    )
                    if self.support_experts['enabled']
                    else self.inductiveBaseGateValues(compound_indices)
                )
            ),
            cold_herb_protein_weight=self.weight.get(
                'context_herb_protein_cold'
            ),
            cold_herb_protein_scale=self.supportExpertColdGateValues(
                compound_indices
            ),
            herb_disease_scale=(
                support_state_gates['herb_disease']
                if support_state_gates is not None else None
            ),
            herb_prototype_residual=self.herbPrototypePairResiduals(
                compound_indices, protein_indices
            ),
            herb_prototype_scale=self.weight.get('herb_prototype_scale'),
            herb_jaccard_residual=self.herbJaccardPairResiduals(
                compound_indices, protein_indices
            ) if (
                self.herb_prototype_transfer['dual_granularity']
                and self.herb_prototype_transfer['dual_mode']
                == 'independent_scales'
            ) else None,
            herb_jaccard_scale=self.weight.get('herb_jaccard_scale'),
        )

    def predictForRanking(self):
        print('hdctipredict----------------------------------------------------------------------------')
        decoder_type = self.pair_decoder['type']
        pairwise_prediction_required = (
            decoder_type == 'mlp'
            or self.herb_context_attention['mode'] in TARGET_HERB_ATTENTION_MODES
            or self.support_state_routing['enabled']
            or self.herb_prototype_transfer['enabled']
            or self.safe_listwise_reranker['enabled']
        )
        if pairwise_prediction_required:
            scores = np.empty((self.num_compounds, self.num_proteins), dtype=np.float32)
            batch_size = self.pair_decoder['prediction_batch_size']
            total_pairs = self.num_compounds * self.num_proteins
            for start in range(0, total_pairs, batch_size):
                stop = min(start + batch_size, total_pairs)
                flat_indices = np.arange(start, stop, dtype=np.int64)
                compound_indices = flat_indices // self.num_proteins
                protein_indices = flat_indices % self.num_proteins
                scores.reshape(-1)[start:stop] = self.predictForPairs(
                    compound_indices, protein_indices
                )
            return scores
        if decoder_type == 'dot':
            scores = self.u.dot(self.i.transpose())
        elif decoder_type == 'bilinear':
            scores = (self.u.dot(self.weight['decoder_bilinear'])).dot(self.i.transpose())
        else:
            raise ValueError('Unsupported pair decoder: %s' % decoder_type)

        support_expert_cold_gates = None
        if self.support_experts['enabled']:
            support_expert_cold_gates = self.supportExpertColdGateValues(
                np.arange(self.num_compounds, dtype=np.int64)
            )
            scores *= (1.0 - support_expert_cold_gates)[:, None]

        if self.inductive_context['suppress_base_zero_support']:
            base_gates = self.inductiveBaseGateValues(
                np.arange(self.num_compounds, dtype=np.int64)
            )
            scores *= base_gates[:, None]

        if self.context_terms['compound_disease']:
            scores += (self.u * self.weight['context_compound_disease']).dot(
                self.i_context.transpose()
            )
        if self.context_terms['herb_protein']:
            herb_protein_scores = (
                self.u_context * self.weight['context_herb_protein']
            ).dot(
                self.i.transpose()
            )
            if self.support_router['enabled']:
                support_gates = self.supportContextGateValues(
                    {
                        'weights': self.weight,
                    },
                    np.arange(self.num_compounds, dtype=np.int64),
                )
                herb_protein_scores *= support_gates[:, None]
            if self.support_experts['enabled']:
                herb_protein_scores *= (
                    1.0 - support_expert_cold_gates
                )[:, None]
            scores += herb_protein_scores
            if self.support_experts['enabled']:
                cold_herb_protein_scores = (
                    self.u_context
                    * self.weight['context_herb_protein_cold']
                ).dot(self.i.transpose())
                scores += (
                    cold_herb_protein_scores
                    * support_expert_cold_gates[:, None]
                )
        if self.context_terms['herb_disease']:
            scores += (self.u_context * self.weight['context_herb_disease']).dot(
                self.i_context.transpose()
            )
        return scores
