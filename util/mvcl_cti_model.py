"""MVCL-CTI dense reference and sparse execution path; see adaptation notice.

Dense execution is guarded; sparse execution requires blockwise losses.
No SACTI modules, donor scores or held-out labels enter this model.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from util.mvcl_cti_data import NODE_TYPES
from util.mvcl_features import fixed_node_features
from util.mvcl_torch_metapath import TorchBooleanMetapath
from util.mvcl_block_ops import block_attention, block_identity_contrast, block_reconstruction
from util.mvcl_sparse_graph import torch_sparse, row_mean, homogeneous
from util.mvcl_cti_data import matrix_hash
import hashlib
import json

RELATIONS = {'H_C': ('herb', 'compound'), 'C_H': ('compound', 'herb'),
             'C_P': ('compound', 'protein'), 'P_C': ('protein', 'compound'),
             'P_D': ('protein', 'disease'), 'D_P': ('disease', 'protein')}
PATHS = {'CHC': 'compound', 'CPC': 'compound', 'PDP': 'protein', 'PCP': 'protein'}


def unit(x):
    return x / (torch.linalg.vector_norm(x, dim=1, keepdim=True) + 1e-12)


class ViewAttention(nn.Module):
    def __init__(self, dimension, hidden=64):
        super().__init__()
        self.first = nn.Linear(dimension, hidden)
        self.second = nn.Linear(hidden, 1, bias=False)

    def forward(self, views):
        if not views[0].shape[0]:
            return views[0]
        logits = torch.stack([self.second(torch.tanh(self.first(v)).mean(0)).squeeze(-1)
                              for v in views])
        return (torch.stack(views) * logits.softmax(0)[:, None, None]).sum(0)


class RelationLayer(nn.Module):
    def __init__(self, dimension, attentive=False):
        super().__init__()
        self.transforms = nn.ModuleDict({r: nn.Linear(dimension, dimension) for r in RELATIONS})
        self.attentive = attentive
        if attentive:
            self.attention = nn.ModuleDict({k: nn.Sequential(nn.Linear(dimension, 128), nn.ReLU(),
                                                           nn.Linear(128, 1)) for k in NODE_TYPES})

    def forward(self, states, relations):
        current, result = dict(states), {}
        # Explicit counterpart of the upstream diffusion update sequence.
        for kind in ('disease', 'compound', 'protein', 'herb'):
            messages = [states[kind]]
            for relation, (target, source) in RELATIONS.items():
                if target == kind:
                    matrix = relations[relation]
                    inputs = current[source] if self.attentive else states[source]
                    messages.append(row_mean(matrix, F.relu(self.transforms[relation](inputs))))
            stack = torch.stack(messages, dim=1)
            value = stack.mean(1)
            if self.attentive:
                weights = self.attention[kind](stack).softmax(1)
                value = value + (weights * stack).sum(1)
                current[kind] = value
            result[kind] = value
        return result


class PathLayer(nn.Module):
    def __init__(self, dimension, dropout):
        super().__init__()
        self.project = nn.ModuleDict({p: nn.Linear(dimension, dimension, bias=False) for p in PATHS})
        self.gamma = nn.ParameterDict({p: nn.Parameter(torch.ones(dimension)) for p in PATHS})
        self.beta = nn.ParameterDict({p: nn.Parameter(torch.zeros(dimension)) for p in PATHS})
        self.bias = nn.ParameterDict({p: nn.Parameter(torch.zeros(dimension)) for p in PATHS})
        self.activate = nn.ModuleDict({p: nn.PReLU() for p in PATHS})
        self.semantic_fc = nn.ModuleDict({k: nn.Linear(dimension, dimension) for k in ('compound', 'protein')})
        self.semantic_vector = nn.ParameterDict({k: nn.Parameter(torch.randn(dimension) * .1)
                                                 for k in ('compound', 'protein')})
        self.dropout = nn.Dropout(dropout)

    def forward(self, states, operators):
        result = dict(states)
        for kind in ('compound', 'protein'):
            views = []
            for path, target in PATHS.items():
                if target == kind:
                    projected = self.project[path](states[kind])
                    value = operators[path](projected) * self.gamma[path] + self.beta[path]
                    views.append(self.activate[path](value + self.bias[path] + projected))
            if not states[kind].shape[0]:
                continue
            vector = self.dropout(self.semantic_vector[kind])
            logits = torch.stack([(torch.tanh(self.semantic_fc[kind](v)).mean(0) * vector).sum()
                                  for v in views])
            result[kind] = (torch.stack(views) * logits.softmax(0)[:, None, None]).sum(0) + states[kind]
        return result


class CrossViewContrast(nn.Module):
    def __init__(self, dimension, tau=.5, block_size=None):
        super().__init__()
        if tau <= 0:
            raise ValueError('Positive contrast temperature required.')
        self.tau = tau
        self.block_size = block_size
        self.q, self.k, self.v = (nn.Linear(dimension, dimension) for _ in range(3))
        self.project = nn.ModuleDict({kind: nn.Sequential(nn.Linear(dimension, dimension), nn.ELU(),
                                                         nn.Linear(dimension, dimension))
                                      for kind in ('compound', 'protein')})

    def attend(self, x):
        if self.block_size is not None:
            return block_attention(self.q(x), self.k(x), self.v(x), self.block_size)
        return (self.q(x) @ self.k(x).T / math.sqrt(x.shape[1])).softmax(1) @ self.v(x)

    def forward(self, left, right):
        loss = left['compound'].sum() * 0
        for kind in ('compound', 'protein'):
            if not len(left[kind]):
                continue
            a, b = (unit(self.project[kind](self.attend(view[kind]))) for view in (left, right))
            if self.block_size is not None:
                loss = loss + block_identity_contrast(a, b, self.tau, self.block_size)
                continue
            similarity = torch.exp(a @ b.T / self.tau)
            # Match upstream exp(cos/tau) THEN softmax, with identity positives.
            loss = loss - F.log_softmax(similarity, dim=1).diagonal().mean()
            loss = loss - F.log_softmax(similarity.T, dim=1).diagonal().mean()
        return loss


class MVCLCTIReference(nn.Module):
    def __init__(self, data, dimension=16, seed=52026, path_dropout=.2, edge_dropout=.5,
                 diffusion_steps=3, restart=.9, block_size=None, sparse_graph=False):
        super().__init__()
        counts = {k: len(data['ids'][k]) for k in NODE_TYPES}
        if sum(counts.values()) > 1024 and not sparse_graph:
            raise ValueError('Small-graph reference limited to 1024 total nodes; real-data runner unavailable.')
        if sparse_graph and block_size is None:
            raise ValueError('Sparse graph requires blockwise losses.')
        if dimension < 1 or diffusion_steps < 0 or not 0 <= restart <= 1:
            raise ValueError('Invalid model dimensions or diffusion settings.')
        self.counts, self.steps, self.restart = counts, diffusion_steps, restart
        if block_size is not None and (not isinstance(block_size, int) or block_size < 1):
            raise ValueError('block_size must be a positive integer.')
        self.block_size = block_size
        self.sparse_graph = sparse_graph
        self._relations_cpu = {r: m.copy().tocsr() for r, m in data['relations'].items()}
        self._mask_cpu = data['train_mask'].copy().tocsr()
        self.edge_dropout = nn.Dropout(edge_dropout)
        features, self.feature_metadata = fixed_node_features(data['ids'])
        for kind in NODE_TYPES:
            self.register_buffer('features_' + kind, torch.from_numpy(features[kind]))
        for relation, matrix in data['relations'].items():
            self.register_buffer('relation_' + relation, torch_sparse(matrix) if sparse_graph
                                 else torch.tensor(matrix.toarray(), dtype=torch.float32))
        self.register_buffer('train_mask', torch_sparse(data['train_mask']) if sparse_graph
                             else torch.tensor(data['train_mask'].toarray(), dtype=torch.float32))
        self.operators = {p: TorchBooleanMetapath(data['relations'][r]) for p, r in
                          dict(CHC='C_H', CPC='C_P', PDP='P_D', PCP='P_C').items()}
        adjacency = homogeneous(self._relations_cpu, RELATIONS, NODE_TYPES, counts)
        self.register_buffer('adjacency', torch_sparse(adjacency) if sparse_graph
                             else torch.tensor(adjacency.toarray(), dtype=torch.float32))
        self.checkpoint_identity = dict(
            version='mvcl_cti_sparse_v1', ids=data['ids'], features=self.feature_metadata,
            relations={r: matrix_hash(m) for r, m in self._relations_cpu.items()},
            train_mask=matrix_hash(self._mask_cpu),
            protocol={k: data.get('report', {}).get(k) for k in
                      ('train_sha256', 'validation_sha256', 'source_hashes', 'fold')},
            model=dict(dimension=dimension, path_dropout=path_dropout, edge_dropout=edge_dropout,
                       diffusion_steps=diffusion_steps, restart=restart, seed=seed,
                       block_size=block_size, sparse_graph=sparse_graph))
        self.checkpoint_identity = json.loads(json.dumps(self.checkpoint_identity, sort_keys=True))
        self.checkpoint_fingerprint = hashlib.sha256(
            json.dumps(self.checkpoint_identity, sort_keys=True).encode()).hexdigest()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.input_projection = nn.ModuleDict({k: nn.Linear(128, dimension) for k in NODE_TYPES})
            self.neighbor = nn.ModuleList([RelationLayer(dimension) for _ in range(2)])
            self.metapath = nn.ModuleList([PathLayer(dimension, path_dropout) for _ in range(2)])
            self.diffusion = nn.ModuleList([RelationLayer(dimension, True) for _ in range(2)])
            self.fusion = ViewAttention(dimension)
            self.contrast = CrossViewContrast(dimension, block_size=block_size)
            self.decoder = nn.ParameterDict({r: nn.Parameter(torch.diag(torch.randn(dimension) * .1))
                                              for r in ('H_C', 'C_P', 'P_D')})
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, 0, .1)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, .1)

    def propagate(self, states):
        x = torch.cat([states[k] for k in NODE_TYPES])
        initial = x
        degree = (torch.sparse.sum(self.adjacency, dim=1).to_dense() if self.sparse_graph
                  else self.adjacency.sum(1))
        norm = degree.clamp_min(1e-12).rsqrt()
        for _ in range(self.steps):
            if self.sparse_graph:
                matrix = torch.sparse_coo_tensor(self.adjacency.indices(),
                    self.edge_dropout(self.adjacency.values()), self.adjacency.shape).coalesce()
                propagated = torch.sparse.mm(matrix, x * norm[:, None])
            else:
                matrix = self.edge_dropout(self.adjacency)
                propagated = matrix @ (x * norm[:, None])
            x = (1-self.restart) * propagated * norm[:, None] + self.restart * initial
        return dict(zip(NODE_TYPES, torch.split(x, [self.counts[k] for k in NODE_TYPES])))

    def encode(self):
        relations = {r: getattr(self, 'relation_' + r) for r in RELATIONS}
        initial = {k: self.input_projection[k](getattr(self, 'features_' + k)) for k in NODE_TYPES}
        neighbor, metapath, diffusion = initial, initial, initial
        for layer in self.neighbor:
            neighbor = layer(neighbor, relations)
        for layer in self.metapath:
            metapath = layer(metapath, self.operators)
        for layer in self.diffusion:
            diffusion = {k: unit(v) for k, v in self.propagate(layer(diffusion, relations)).items()}
        views = [{k: unit(v) for k, v in view.items()} for view in (neighbor, metapath, diffusion)]
        embeddings = {k: self.fusion([v[k] for v in views]) for k in NODE_TYPES}
        return views, embeddings

    def forward(self, contrast_weight=20000., reg_lambda=.5, return_scores=True):
        if return_scores and sum(self.counts.values()) > 1024:
            raise ValueError('Large graph requires return_scores=False; score requested pairs separately.')
        views, embeddings = self.encode()
        relations = {r: getattr(self, 'relation_' + r) for r in RELATIONS}
        reconstruction = embeddings['compound'].sum() * 0
        scores = {}
        for relation in self.decoder:
            a, b = RELATIONS[relation]
            if self.block_size is not None:
                reconstruction = reconstruction + block_reconstruction(
                    embeddings[a], self.decoder[relation], embeddings[b],
                    self._relations_cpu[relation] if self.sparse_graph else relations[relation],
                    self.block_size, (self._mask_cpu if self.sparse_graph else self.train_mask)
                    if relation == 'C_P' else None)
                continue
            scores[relation] = embeddings[a] @ self.decoder[relation] @ embeddings[b].T
            residual = scores[relation] - relations[relation]
            if relation == 'C_P':
                residual = residual * self.train_mask
            reconstruction = reconstruction + residual.square().sum()
        if return_scores and 'C_P' not in scores:
            scores['C_P'] = embeddings['compound'] @ self.decoder['C_P'] @ embeddings['protein'].T
        contrast = sum(self.contrast(views[a], views[b]) for a, b in ((0, 1), (0, 2), (1, 2)))
        regularization = .5 * sum(p.square().sum() for name, p in self.named_parameters() if 'bias' not in name)
        return dict(scores=scores.get('C_P') if return_scores else None, views=views, embeddings=embeddings,
                    reconstruction=reconstruction, contrast=contrast, regularization=regularization,
                    loss=reconstruction + contrast_weight * contrast + reg_lambda * regularization)
