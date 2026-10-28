"""Explicit relation-size adaptation; the historical MVCL implementation is unchanged."""
import hashlib
import json
import math

from util.mvcl_cti_model import MVCLCTIReference, RELATIONS
from util.mvcl_block_ops import block_reconstruction


class RelationScaledMVCL(MVCLCTIReference):
    def __init__(self, data, reconstruction_scaling='cp_anchored_mean', **kwargs):
        if reconstruction_scaling != 'cp_anchored_mean':
            raise ValueError('Unknown relation scaling policy.')
        super().__init__(data, **kwargs)
        cp_terms = self._mask_cpu.nnz
        if cp_terms < 1:
            raise ValueError('Nonempty C-P training mask required.')
        self.reconstruction_weights = {'C_P': 1.}
        for relation in ('H_C', 'P_D'):
            size = math.prod(self._relations_cpu[relation].shape)
            if size < 1:
                raise ValueError('Nonempty side-relation universe required.')
            self.reconstruction_weights[relation] = cp_terms / size
        self.checkpoint_identity['reconstruction'] = dict(
            policy=reconstruction_scaling, weights=self.reconstruction_weights)
        self.checkpoint_fingerprint = hashlib.sha256(
            json.dumps(self.checkpoint_identity, sort_keys=True).encode()).hexdigest()

    def forward(self, contrast_weight=20000., reg_lambda=.5, return_scores=True):
        if return_scores and sum(self.counts.values()) > 1024:
            raise ValueError('Large graph requires return_scores=False.')
        views, embeddings = self.encode()
        raw = {}
        for relation in self.decoder:
            a, b = RELATIONS[relation]
            raw[relation] = block_reconstruction(
                embeddings[a], self.decoder[relation], embeddings[b],
                self._relations_cpu[relation], self.block_size or 128,
                self._mask_cpu if relation == 'C_P' else None)
        reconstruction = sum(self.reconstruction_weights[r] * value for r, value in raw.items())
        contrast = sum(self.contrast(views[a], views[b]) for a, b in ((0, 1), (0, 2), (1, 2)))
        regularization = .5 * sum(p.square().sum() for n, p in self.named_parameters() if 'bias' not in n)
        scores = (embeddings['compound'] @ self.decoder['C_P'] @ embeddings['protein'].T
                  if return_scores else None)
        return dict(scores=scores, views=views, embeddings=embeddings,
                    reconstruction_by_relation=raw, reconstruction=reconstruction,
                    contrast=contrast, regularization=regularization,
                    loss=reconstruction + contrast_weight * contrast + reg_lambda * regularization)


def build_model(data, **config):
    if 'reconstruction_scaling' in config:
        return RelationScaledMVCL(data, **config)
    return MVCLCTIReference(data, **config)
