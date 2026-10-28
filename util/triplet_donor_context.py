"""Target-conditioned donor weighting using candidate-excluded disease context.

Experimental primitive, not integrated into SACTI or a training runner.
"""

import numpy as np
from scipy import sparse

from util.donor_reweighting import DonorEvidence, binary_matrix, indices


class TripletDonorContext:
    def __init__(self, hc, training_cp, pd, masked_compounds=()):
        self.evidence = DonorEvidence(hc, training_cp, masked_compounds)
        pd = binary_matrix(pd)
        if pd.shape[0] != self.evidence.cp.shape[1]:
            raise ValueError('P-D target order must match C-P columns.')
        degrees = np.asarray(pd.sum(axis=1)).ravel()
        inverse = np.zeros_like(degrees)
        np.divide(1, degrees, out=inverse, where=degrees > 0)
        self.pd = (sparse.diags(inverse) @ pd).tocsr()
        self.profiles = (self.evidence.cp @ self.pd).tocsr()
        self.profile_norm2 = np.asarray(self.profiles.multiply(self.profiles).sum(axis=1)).ravel()
        self.target_norm2 = np.asarray(self.pd.multiply(self.pd).sum(axis=1)).ravel()

    def compatibility(self, donors, targets):
        """Cosine against donor's OTHER targets, in caller-bounded target blocks."""
        indices(donors, self.evidence.cp.shape[0])
        indices(targets, self.evidence.cp.shape[1])
        donors = np.asarray(donors, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.int64)
        product = (self.profiles[donors] @ self.pd[targets].T).toarray()
        observed = self.evidence.cp[donors][:, targets].toarray()
        target_norm2 = self.target_norm2[targets][None, :]
        numerator = product - observed * target_norm2
        remaining_norm2 = (self.profile_norm2[donors, None]
                           - 2 * observed * product + observed * target_norm2)
        # Exact removal of a sole disease-bearing target can leave roundoff dust.
        remaining_norm2 = np.maximum(remaining_norm2, 0)
        denominator = np.sqrt(remaining_norm2 * target_norm2)
        score = np.zeros_like(product)
        np.divide(numerator, denominator, out=score, where=denominator > 1e-12)
        return np.clip(score, 0, 1)

    def score(self, compound, coefficient=0.0, block_size=64):
        if not np.isfinite(coefficient) or not isinstance(block_size, int) or block_size < 1:
            raise ValueError('Invalid conditioning settings.')
        query = self.evidence.query(compound)
        if coefficient == 0 or not len(query.donors):
            return query.score()[0]
        scores = np.zeros(self.evidence.cp.shape[1])
        for start in range(0, len(scores), block_size):
            targets = np.arange(start, min(start + block_size, len(scores)))
            condition = self.compatibility(query.donors, targets)
            logits = np.log(query.jaccard)[:, None] + coefficient * condition
            logits -= np.max(logits, axis=0, keepdims=True)
            weights = np.exp(logits)
            weights /= weights.sum(axis=0, keepdims=True)
            labels = query.labels[:, targets].toarray()
            # Normalize over ALL donors, not only the candidate's supporters.
            scores[targets] = np.sum(weights * labels, axis=0)
        return scores
