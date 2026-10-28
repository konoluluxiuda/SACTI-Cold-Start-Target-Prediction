"""Candidate membership for evaluation only; never used to construct scores."""


def ratio_candidate_proteins(model_proteins, outer_train, outer_test, scope):
    model = {str(value) for value in model_proteins}
    if scope == 'model':
        return sorted(model)
    if scope != 'cp_positive':
        raise ValueError('Unknown ratio candidate scope: ' + scope)
    # Match the original split sampler's global C-P protein universe.
    # Held-out identities define evaluation membership, not model evidence.
    proteins = {str(row[1]) for records in (outer_train, outer_test)
                for row in records if float(row[2]) > 0}
    if not proteins or not proteins.issubset(model):
        raise ValueError('C-P candidate universe is empty or outside model IDs.')
    return sorted(proteins)
