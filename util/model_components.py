import numpy as np


CONTEXT_TERM_CONFIG = {
    'compound_disease': 'context.compound_disease',
    'herb_protein': 'context.herb_protein',
    'herb_disease': 'context.herb_disease',
}

ENCODER_PROFILES = {
    'hdcti': {
        'use_self_gating': True,
        'use_pagerank': True,
        'use_dense_full_attention': True,
        'use_node_dimension_attention': True,
        'external_baseline': False,
    },
    'dual_hgnn_cti': {
        'use_self_gating': False,
        'use_pagerank': False,
        'use_dense_full_attention': False,
        'use_node_dimension_attention': False,
        'external_baseline': True,
    },
    'dual_hgnn_portability': {
        'use_self_gating': False,
        'use_pagerank': False,
        'use_dense_full_attention': False,
        'use_node_dimension_attention': False,
        'external_baseline': False,
    },
}


def _config_bool(config, key, default):
    if not config.contains(key):
        return default
    value = str(config[key]).strip().lower()
    if value in {'1', 'true', 'yes', 'on'}:
        return True
    if value in {'0', 'false', 'no', 'off'}:
        return False
    raise ValueError('Invalid boolean value for %s: %s' % (key, config[key]))


def resolve_encoder_profile(config):
    name = (
        str(config['encoder.profile']).strip().lower()
        if config.contains('encoder.profile') else 'hdcti'
    )
    if name not in ENCODER_PROFILES:
        raise ValueError(
            'encoder.profile must be hdcti, dual_hgnn_cti, or '
            'dual_hgnn_portability.'
        )
    profile = dict(ENCODER_PROFILES[name])
    profile['name'] = name
    return profile


def resolve_context_terms(config):
    master_enabled = _config_bool(config, 'context.interaction', False)
    return {
        term: master_enabled and _config_bool(config, key, True)
        for term, key in CONTEXT_TERM_CONFIG.items()
    }


def resolve_herb_context_attention(config):
    mode = (
        str(config['context.herb_protein.mode']).strip().lower()
        if config.contains('context.herb_protein.mode') else 'static'
    )
    valid_modes = {'static', 'target_attention', 'target_residual_attention'}
    if mode not in valid_modes:
        raise ValueError(
            'context.herb_protein.mode must be static, target_attention, '
            'or target_residual_attention.'
        )
    temperature = (
        float(config['context.herb_attention.temperature'])
        if config.contains('context.herb_attention.temperature') else 1.0
    )
    if temperature <= 0:
        raise ValueError('context.herb_attention.temperature must be positive.')
    return {
        'mode': mode,
        'temperature': temperature,
    }


def resolve_counterfactual_context(config):
    enabled = _config_bool(config, 'counterfactual.context', False)
    settings = {
        'enabled': enabled,
        'weight': (
            float(config['counterfactual.weight'])
            if config.contains('counterfactual.weight') else 0.05
        ),
        'margin': (
            float(config['counterfactual.margin'])
            if config.contains('counterfactual.margin') else 0.2
        ),
        'draws': (
            int(config['counterfactual.draws'])
            if config.contains('counterfactual.draws') else 20
        ),
        'seed': (
            int(config['counterfactual.seed'])
            if config.contains('counterfactual.seed') else 42026
        ),
        'match': (
            str(config['counterfactual.match']).strip().lower()
            if config.contains('counterfactual.match')
            else 'exact_hc_degree_disjoint'
        ),
    }
    if settings['weight'] < 0:
        raise ValueError('counterfactual.weight cannot be negative.')
    if settings['margin'] <= 0:
        raise ValueError('counterfactual.margin must be positive.')
    if settings['draws'] <= 0:
        raise ValueError('counterfactual.draws must be positive.')
    if settings['match'] != 'exact_hc_degree_disjoint':
        raise ValueError(
            'counterfactual.match must be exact_hc_degree_disjoint.'
        )
    if enabled and settings['weight'] == 0:
        raise ValueError(
            'counterfactual.weight must be positive when counterfactual.context=True.'
        )
    return settings


def resolve_context_mask_training(config):
    enabled = _config_bool(config, 'context.mask.training', False)
    side = (
        str(config['context.mask.side']).strip().lower()
        if config.contains('context.mask.side') else 'compound'
    )
    weight = (
        float(config['context.mask.weight'])
        if config.contains('context.mask.weight') else 0.1
    )
    if side not in {'compound', 'protein', 'both'}:
        raise ValueError('context.mask.side must be compound, protein, or both.')
    if weight < 0:
        raise ValueError('context.mask.weight cannot be negative.')
    if enabled and weight == 0:
        raise ValueError(
            'context.mask.weight must be positive when context.mask.training=True.'
        )
    return {
        'enabled': enabled,
        'side': side,
        'weight': weight,
    }


def resolve_support_router(config):
    enabled = _config_bool(config, 'support.router', False)
    mode = (
        str(config['support.router.mode']).strip().lower()
        if config.contains('support.router.mode') else 'monotonic_residual'
    )
    pseudo_cold_ratio = (
        float(config['support.router.pseudo.cold.ratio'])
        if config.contains('support.router.pseudo.cold.ratio') else 0.1
    )
    seed = (
        int(config['support.router.seed'])
        if config.contains('support.router.seed') else 62026
    )
    initial_slope = (
        float(config['support.router.initial.slope'])
        if config.contains('support.router.initial.slope') else 1.0
    )
    if mode != 'monotonic_residual':
        raise ValueError('support.router.mode must be monotonic_residual.')
    if not 0.0 < pseudo_cold_ratio < 1.0:
        raise ValueError(
            'support.router.pseudo.cold.ratio must be between 0 and 1.'
        )
    if initial_slope <= 0:
        raise ValueError('support.router.initial.slope must be positive.')
    return {
        'enabled': enabled,
        'mode': mode,
        'pseudo_cold_ratio': pseudo_cold_ratio,
        'seed': seed,
        'initial_slope': initial_slope,
    }


def resolve_support_experts(config):
    enabled = _config_bool(config, 'support.experts', False)
    mode = (
        str(config['support.experts.mode']).strip().lower()
        if config.contains('support.experts.mode') else 'hard_zero_support'
    )
    pseudo_cold_ratio = (
        float(config['support.experts.pseudo.cold.ratio'])
        if config.contains('support.experts.pseudo.cold.ratio') else 0.1
    )
    seed = (
        int(config['support.experts.seed'])
        if config.contains('support.experts.seed') else 72026
    )
    detach_cold_features = _config_bool(
        config, 'support.experts.detach.cold.features', True
    )
    if mode != 'hard_zero_support':
        raise ValueError(
            'support.experts.mode must be hard_zero_support.'
        )
    if not 0.0 < pseudo_cold_ratio < 1.0:
        raise ValueError(
            'support.experts.pseudo.cold.ratio must be between 0 and 1.'
        )
    if enabled and not detach_cold_features:
        raise ValueError(
            'The frozen SCHE Pilot requires '
            'support.experts.detach.cold.features=True.'
        )
    return {
        'enabled': enabled,
        'mode': mode,
        'pseudo_cold_ratio': pseudo_cold_ratio,
        'seed': seed,
        'detach_cold_features': detach_cold_features,
    }


def resolve_support_state_routing(config):
    enabled = _config_bool(config, 'support.state.routing', False)
    mode = (
        str(config['support.state.routing.mode']).strip().lower()
        if config.contains('support.state.routing.mode')
        else 'hard_four_state'
    )
    training_mode = (
        str(config['support.state.routing.training']).strip().lower()
        if config.contains('support.state.routing.training')
        else 'joint_ww'
    )
    herb_protein_auxiliary_weight = (
        float(config['support.state.routing.herb_protein.aux.weight'])
        if config.contains(
            'support.state.routing.herb_protein.aux.weight'
        )
        else 1.0
    )
    auxiliary_weight = (
        float(config['support.state.routing.cold_cold.aux.weight'])
        if config.contains('support.state.routing.cold_cold.aux.weight')
        else 1.0
    )
    detach_cold_cold_features = _config_bool(
        config,
        'support.state.routing.detach.cold_cold.features',
        True,
    )
    if mode != 'hard_four_state':
        raise ValueError(
            'support.state.routing.mode must be hard_four_state.'
        )
    if training_mode not in {'joint_ww', 'isolated_heads'}:
        raise ValueError(
            'support.state.routing.training must be joint_ww or '
            'isolated_heads.'
        )
    if herb_protein_auxiliary_weight < 0:
        raise ValueError(
            'support.state.routing.herb_protein.aux.weight cannot be '
            'negative.'
        )
    if auxiliary_weight < 0:
        raise ValueError(
            'support.state.routing.cold_cold.aux.weight cannot be negative.'
        )
    if enabled and auxiliary_weight == 0:
        raise ValueError(
            'support.state.routing.cold_cold.aux.weight must be positive '
            'when support.state.routing=True.'
        )
    if (
        enabled
        and training_mode == 'isolated_heads'
        and herb_protein_auxiliary_weight == 0
    ):
        raise ValueError(
            'support.state.routing.herb_protein.aux.weight must be '
            'positive for isolated_heads training.'
        )
    if enabled and not detach_cold_cold_features:
        raise ValueError(
            'support.state.routing requires detached cold-cold features.'
        )
    return {
        'enabled': enabled,
        'mode': mode,
        'training_mode': training_mode,
        'herb_protein_aux_weight': herb_protein_auxiliary_weight,
        'cold_cold_aux_weight': auxiliary_weight,
        'detach_cold_cold_features': detach_cold_cold_features,
    }


def support_state_pair_gates(
        compound_support_degrees,
        protein_support_degrees,
        compound_context_available,
        protein_context_available):
    compound_support_degrees = np.asarray(
        compound_support_degrees, dtype=np.float64
    )
    protein_support_degrees = np.asarray(
        protein_support_degrees, dtype=np.float64
    )
    compound_context_available = np.asarray(
        compound_context_available, dtype=np.float64
    )
    protein_context_available = np.asarray(
        protein_context_available, dtype=np.float64
    )
    expected_shape = compound_support_degrees.shape
    for name, values in (
        ('protein support degrees', protein_support_degrees),
        ('compound context availability', compound_context_available),
        ('protein context availability', protein_context_available),
    ):
        if values.shape != expected_shape:
            raise ValueError(
                '%s must match compound support degree shape.' % name
            )

    compound_warm = compound_support_degrees > 0
    protein_warm = protein_support_degrees > 0
    compound_context = compound_context_available > 0
    protein_context = protein_context_available > 0
    return {
        'base': compound_warm.astype(np.float32),
        'herb_protein': (
            protein_warm & compound_context
        ).astype(np.float32),
        'herb_disease': (
            ~compound_warm
            & ~protein_warm
            & compound_context
            & protein_context
        ).astype(np.float32),
        'state': np.select(
            [
                compound_warm & protein_warm,
                ~compound_warm & protein_warm,
                compound_warm & ~protein_warm,
            ],
            [0, 1, 2],
            default=3,
        ).astype(np.int32),
    }


def resolve_inductive_context(config):
    enabled = _config_bool(config, 'inductive.context', False)
    suppress_base = _config_bool(
        config, 'inductive.context.suppress.base.zero.support', True
    )
    self_excluded = _config_bool(
        config, 'inductive.context.self.excluded', True
    )
    if enabled and not (suppress_base or self_excluded):
        raise ValueError(
            'inductive.context requires base suppression, self-excluded '
            'context, or both.'
        )
    return {
        'enabled': enabled,
        'suppress_base_zero_support': enabled and suppress_base,
        'self_excluded': enabled and self_excluded,
    }


def support_decoupled_base_gate(support_degrees, context_available):
    support_degrees = np.asarray(support_degrees, dtype=np.float64)
    context_available = np.asarray(context_available, dtype=np.float64)
    if support_degrees.shape != context_available.shape:
        raise ValueError(
            'Support degrees and context availability must have matching shapes.'
        )
    suppress = (support_degrees <= 0) & (context_available > 0)
    return np.where(suppress, 0.0, 1.0).astype(np.float32)


def support_conditioned_cold_gate(support_degrees, context_available):
    support_degrees = np.asarray(support_degrees, dtype=np.float64)
    context_available = np.asarray(context_available, dtype=np.float64)
    if support_degrees.shape != context_available.shape:
        raise ValueError(
            'Support degrees and context availability must have matching shapes.'
        )
    use_cold = (support_degrees <= 0) & (context_available > 0)
    return use_cold.astype(np.float32)


def resolve_hyperedge_attention(config):
    enabled = _config_bool(config, 'hyperedge.attention', False)
    mode = (
        str(config['hyperedge.attention.mode']).strip().lower()
        if config.contains('hyperedge.attention.mode')
        else 'factorized_specificity'
    )
    hc_enabled = _config_bool(config, 'hyperedge.attention.hc', True)
    pd_enabled = _config_bool(config, 'hyperedge.attention.pd', True)
    temperature = (
        float(config['hyperedge.attention.temperature'])
        if config.contains('hyperedge.attention.temperature') else 1.0
    )
    prior_scale = (
        float(config['hyperedge.attention.prior.scale'])
        if config.contains('hyperedge.attention.prior.scale') else 0.1
    )
    if mode != 'factorized_specificity':
        raise ValueError(
            'hyperedge.attention.mode must be factorized_specificity.'
        )
    if enabled and not (hc_enabled or pd_enabled):
        raise ValueError(
            'At least one of hyperedge.attention.hc/pd must be enabled.'
        )
    if temperature <= 0:
        raise ValueError('hyperedge.attention.temperature must be positive.')
    if prior_scale < 0:
        raise ValueError(
            'hyperedge.attention.prior.scale cannot be negative.'
        )
    return {
        'enabled': enabled,
        'mode': mode,
        'hc_enabled': enabled and hc_enabled,
        'pd_enabled': enabled and pd_enabled,
        'temperature': temperature,
        'prior_scale': prior_scale,
    }


def resolve_global_token_attention(config):
    enabled = _config_bool(config, 'global.token.attention', False)
    mode = (
        str(config['global.token.attention.mode']).strip().lower()
        if config.contains('global.token.attention.mode')
        else 'hyperedge_induced'
    )
    hc_enabled = _config_bool(config, 'global.token.attention.hc', True)
    pd_enabled = _config_bool(config, 'global.token.attention.pd', True)
    token_count = (
        int(config['global.token.attention.tokens'])
        if config.contains('global.token.attention.tokens') else 32
    )
    head_count = (
        int(config['global.token.attention.heads'])
        if config.contains('global.token.attention.heads') else 2
    )
    temperature = (
        float(config['global.token.attention.temperature'])
        if config.contains('global.token.attention.temperature') else 1.0
    )
    if mode != 'hyperedge_induced':
        raise ValueError(
            'global.token.attention.mode must be hyperedge_induced.'
        )
    if enabled and not (hc_enabled or pd_enabled):
        raise ValueError(
            'At least one of global.token.attention.hc/pd must be enabled.'
        )
    if token_count <= 0:
        raise ValueError('global.token.attention.tokens must be positive.')
    if head_count <= 0:
        raise ValueError('global.token.attention.heads must be positive.')
    if temperature <= 0:
        raise ValueError(
            'global.token.attention.temperature must be positive.'
        )
    return {
        'enabled': enabled,
        'mode': mode,
        'hc_enabled': enabled and hc_enabled,
        'pd_enabled': enabled and pd_enabled,
        'tokens': token_count,
        'heads': head_count,
        'temperature': temperature,
    }


def resolve_hplga(config):
    enabled = _config_bool(config, 'hplga.enabled', False)
    mode = (
        str(config['hplga.mode']).strip().lower()
        if config.contains('hplga.mode')
        else 'hypergraph_pagerank_linear'
    )
    kernel = (
        str(config['hplga.kernel']).strip().lower()
        if config.contains('hplga.kernel') else 'elu_plus_one'
    )
    hc_enabled = _config_bool(config, 'hplga.hc', True)
    pd_enabled = _config_bool(config, 'hplga.pd', True)
    head_count = (
        int(config['hplga.heads'])
        if config.contains('hplga.heads') else 2
    )
    pagerank_alpha = (
        float(config['hplga.pagerank.alpha'])
        if config.contains('hplga.pagerank.alpha') else 0.85
    )
    pagerank_max_iter = (
        int(config['hplga.pagerank.max.iter'])
        if config.contains('hplga.pagerank.max.iter') else 100
    )
    pagerank_tol = (
        float(config['hplga.pagerank.tol'])
        if config.contains('hplga.pagerank.tol') else 1e-8
    )
    epsilon = (
        float(config['hplga.epsilon'])
        if config.contains('hplga.epsilon') else 1e-6
    )

    if mode != 'hypergraph_pagerank_linear':
        raise ValueError(
            'hplga.mode must be hypergraph_pagerank_linear.'
        )
    if kernel != 'elu_plus_one':
        raise ValueError('hplga.kernel must be elu_plus_one.')
    if enabled and not (hc_enabled or pd_enabled):
        raise ValueError('At least one of hplga.hc/pd must be enabled.')
    if head_count <= 0:
        raise ValueError('hplga.heads must be positive.')
    if not 0.0 < pagerank_alpha < 1.0:
        raise ValueError('hplga.pagerank.alpha must be between 0 and 1.')
    if pagerank_max_iter <= 0:
        raise ValueError('hplga.pagerank.max.iter must be positive.')
    if pagerank_tol <= 0:
        raise ValueError('hplga.pagerank.tol must be positive.')
    if epsilon <= 0:
        raise ValueError('hplga.epsilon must be positive.')

    return {
        'enabled': enabled,
        'mode': mode,
        'kernel': kernel,
        'hc_enabled': enabled and hc_enabled,
        'pd_enabled': enabled and pd_enabled,
        'heads': head_count,
        'pagerank_alpha': pagerank_alpha,
        'pagerank_max_iter': pagerank_max_iter,
        'pagerank_tol': pagerank_tol,
        'epsilon': epsilon,
    }


def counterfactual_margin_values(
        factual_context_logits,
        counterfactual_context_logits,
        labels,
        eligible_mask,
        margin):
    factual = np.asarray(factual_context_logits, dtype=np.float64)
    counterfactual = np.asarray(counterfactual_context_logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    eligible = np.asarray(eligible_mask, dtype=np.float64)
    if not (
            factual.shape == counterfactual.shape == labels.shape
            == eligible.shape):
        raise ValueError('Counterfactual margin inputs must have matching shapes.')
    active = (labels > 0.5).astype(np.float64) * (eligible > 0).astype(np.float64)
    values = np.maximum(
        0.0, float(margin) - (factual - counterfactual)
    ) * active
    return values


def resolve_early_stopping(config):
    enabled = _config_bool(config, 'early.stopping', False)
    settings = {
        'enabled': enabled,
        'ratio': float(config['validation.ratio']) if config.contains('validation.ratio') else 0.1,
        'metric': str(config['validation.metric']).strip().lower()
        if config.contains('validation.metric') else 'aupr',
        'interval': int(config['validation.interval'])
        if config.contains('validation.interval') else 2,
        'patience': int(config['validation.patience'])
        if config.contains('validation.patience') else 5,
        'min_delta': float(config['validation.min.delta'])
        if config.contains('validation.min.delta') else 0.0001,
    }
    if not 0.0 < settings['ratio'] < 1.0:
        raise ValueError('validation.ratio must be between 0 and 1.')
    if settings['metric'] not in {'aupr', 'auc'}:
        raise ValueError('validation.metric must be AUPR or AUC.')
    if settings['interval'] <= 0:
        raise ValueError('validation.interval must be positive.')
    if settings['patience'] <= 0:
        raise ValueError('validation.patience must be positive.')
    if settings['min_delta'] < 0:
        raise ValueError('validation.min.delta cannot be negative.')
    return settings


def requires_inner_validation(config, early_stopping=None):
    """Return whether orchestration must reserve a fold-local validation set."""
    settings = early_stopping or resolve_early_stopping(config)
    return (
        settings['enabled']
        or _config_bool(config, 'safe.listwise.reranker', False)
    )


def resolve_negative_sampling(config):
    strategy = (
        str(config['negative.strategy']).strip().lower()
        if config.contains('negative.strategy') else 'random'
    )
    if strategy not in {'random', 'mixed'}:
        raise ValueError('negative.strategy must be random or mixed.')
    hard_ratio = (
        float(config['negative.hard.ratio'])
        if config.contains('negative.hard.ratio') else 0.25
    )
    if not 0.0 <= hard_ratio <= 1.0:
        raise ValueError('negative.hard.ratio must be between 0 and 1.')
    if strategy == 'mixed' and hard_ratio == 0.0:
        raise ValueError('negative.hard.ratio must be positive for mixed sampling.')
    return {
        'strategy': strategy,
        'hard_ratio': hard_ratio,
    }


def resolve_pair_decoder(config):
    decoder_type = (
        str(config['pair.decoder']).strip().lower()
        if config.contains('pair.decoder') else 'dot'
    )
    if decoder_type not in {'dot', 'bilinear', 'mlp'}:
        raise ValueError('pair.decoder must be dot, bilinear, or mlp.')
    hidden_size = (
        int(config['pair.mlp.hidden'])
        if config.contains('pair.mlp.hidden') else 64
    )
    prediction_batch_size = (
        int(config['pair.prediction.batch.size'])
        if config.contains('pair.prediction.batch.size') else 65536
    )
    if hidden_size <= 0:
        raise ValueError('pair.mlp.hidden must be positive.')
    if prediction_batch_size <= 0:
        raise ValueError('pair.prediction.batch.size must be positive.')
    return {
        'type': decoder_type,
        'hidden_size': hidden_size,
        'prediction_batch_size': prediction_batch_size,
    }


def resolve_herb_prototype_transfer(config):
    enabled = _config_bool(config, 'herb.prototype.transfer', False)
    mode = (
        str(config['herb.prototype.mode']).strip().lower()
        if config.contains('herb.prototype.mode')
        else 'support_calibrated_loco'
    )
    prior_strength = (
        float(config['herb.prototype.prior'])
        if config.contains('herb.prototype.prior') else 1.0
    )
    replace_compound_pagerank = _config_bool(
        config, 'herb.prototype.replace.compound.pagerank', enabled
    )
    dual_granularity = _config_bool(
        config, 'herb.prototype.dual.granularity', False
    )
    prototype_component = _config_bool(
        config, 'herb.prototype.prototype.enabled', True
    )
    dual_mode = (
        str(config['herb.prototype.dual.mode']).strip().lower()
        if config.contains('herb.prototype.dual.mode') else 'fixed_blend'
    )
    jaccard_blend = (
        float(config['herb.prototype.jaccard.blend'])
        if config.contains('herb.prototype.jaccard.blend') else 0.5
    )
    if mode != 'support_calibrated_loco':
        raise ValueError(
            'herb.prototype.mode must be support_calibrated_loco.'
        )
    if prior_strength <= 0:
        raise ValueError('herb.prototype.prior must be positive.')
    if not enabled and replace_compound_pagerank:
        raise ValueError(
            'Compound PageRank replacement requires herb.prototype.transfer=True.'
        )
    if dual_granularity and not enabled:
        raise ValueError(
            'Dual-granularity transfer requires herb.prototype.transfer=True.'
        )
    if enabled and not prototype_component and not dual_granularity:
        raise ValueError(
            'Disabling the prototype component requires dual-granularity '
            'Jaccard transfer.'
        )
    if (
        enabled and not prototype_component
        and dual_mode != 'independent_scales'
    ):
        raise ValueError(
            'Jaccard-only transfer requires independent_scales mode.'
        )
    if dual_mode not in {'fixed_blend', 'independent_scales'}:
        raise ValueError(
            'herb.prototype.dual.mode must be fixed_blend or '
            'independent_scales.'
        )
    if jaccard_blend < 0 or jaccard_blend > 1:
        raise ValueError('herb.prototype.jaccard.blend must be in [0, 1].')
    return {
        'enabled': enabled,
        'mode': mode,
        'prior_strength': prior_strength,
        'replace_compound_pagerank': replace_compound_pagerank,
        'prototype_component': prototype_component,
        'dual_granularity': dual_granularity,
        'dual_mode': dual_mode,
        'jaccard_blend': jaccard_blend,
    }


def resolve_hc_ranking_distillation(config, herb_prototype_transfer=None):
    enabled = _config_bool(config, 'hc.ranking.distillation', False)
    settings = {
        'enabled': enabled,
        'weight': float(config['hc.ranking.distillation.weight'])
        if config.contains('hc.ranking.distillation.weight') else 0.1,
        'temperature': float(config['hc.ranking.distillation.temperature'])
        if config.contains('hc.ranking.distillation.temperature') else 0.5,
        'queries_per_update': int(
            config['hc.ranking.distillation.queries.per.update']
        ) if config.contains(
            'hc.ranking.distillation.queries.per.update'
        ) else 32,
        'hard_negatives': int(
            config['hc.ranking.distillation.hard.negatives']
        ) if config.contains('hc.ranking.distillation.hard.negatives') else 8,
        'random_negatives': int(
            config['hc.ranking.distillation.random.negatives']
        ) if config.contains('hc.ranking.distillation.random.negatives') else 8,
        'hard_pool_size': int(
            config['hc.ranking.distillation.hard.pool.size']
        ) if config.contains('hc.ranking.distillation.hard.pool.size') else 64,
        'interval': int(config['hc.ranking.distillation.interval'])
        if config.contains('hc.ranking.distillation.interval') else 4,
        'seed_offset': int(config['hc.ranking.distillation.seed.offset'])
        if config.contains('hc.ranking.distillation.seed.offset') else 730000,
    }
    positive_keys = (
        'weight', 'temperature', 'queries_per_update', 'hard_pool_size',
        'interval',
    )
    for key in positive_keys:
        if settings[key] <= 0:
            raise ValueError('hc.ranking.distillation.%s must be positive.' % key)
    if settings['hard_negatives'] < 0 or settings['random_negatives'] < 0:
        raise ValueError('Distillation negative counts cannot be negative.')
    negative_count = settings['hard_negatives'] + settings['random_negatives']
    if negative_count <= 0:
        raise ValueError('Distillation requires at least one negative.')
    if settings['hard_pool_size'] < negative_count:
        raise ValueError(
            'Distillation hard pool must cover hard and random negatives.'
        )
    settings['candidate_count'] = 1 + negative_count
    if enabled:
        transfer = herb_prototype_transfer or resolve_herb_prototype_transfer(
            config
        )
        if not transfer['enabled']:
            raise ValueError(
                'H-C ranking distillation requires herb prototype transfer.'
            )
        if not transfer['dual_granularity']:
            raise ValueError(
                'H-C ranking distillation requires dual-granularity transfer.'
            )
        if transfer['dual_mode'] != 'independent_scales':
            raise ValueError(
                'H-C ranking distillation requires independent prototype and '
                'Jaccard scales.'
            )
    return settings


def resolve_safe_listwise_reranker(
        config, herb_prototype_transfer=None, hc_ranking_distillation=None):
    enabled = _config_bool(config, 'safe.listwise.reranker', False)
    settings = {
        'enabled': enabled,
        'base_expert': (
            str(config['safe.listwise.base.expert']).strip().lower()
            if config.contains('safe.listwise.base.expert') else 'prototype'
        ),
        'tau': float(config['safe.listwise.tau'])
        if config.contains('safe.listwise.tau') else 0.02,
        'hidden_size': int(config['safe.listwise.hidden'])
        if config.contains('safe.listwise.hidden') else 64,
        'gate_hidden_size': int(config['safe.listwise.gate.hidden'])
        if config.contains('safe.listwise.gate.hidden') else 8,
        'queries_per_update': int(config['safe.listwise.queries.per.update'])
        if config.contains('safe.listwise.queries.per.update') else 32,
        'hard_unlabeled': int(config['safe.listwise.hard.unlabeled'])
        if config.contains('safe.listwise.hard.unlabeled') else 256,
        'random_unlabeled': int(config['safe.listwise.random.unlabeled'])
        if config.contains('safe.listwise.random.unlabeled') else 64,
        'interval': int(config['safe.listwise.interval'])
        if config.contains('safe.listwise.interval') else 1,
        'temperature': float(config['safe.listwise.temperature'])
        if config.contains('safe.listwise.temperature') else 1.0,
        'listwise_weight': float(config['safe.listwise.loss.listwise'])
        if config.contains('safe.listwise.loss.listwise') else 1.0,
        'pairwise_weight': float(config['safe.listwise.loss.pairwise'])
        if config.contains('safe.listwise.loss.pairwise') else 0.25,
        'bce_weight': float(config['safe.listwise.loss.bce'])
        if config.contains('safe.listwise.loss.bce') else 0.1,
        'trust_weight': float(config['safe.listwise.loss.trust'])
        if config.contains('safe.listwise.loss.trust') else 0.1,
        'seed_offset': int(config['safe.listwise.seed.offset'])
        if config.contains('safe.listwise.seed.offset') else 831026,
        'validation_interval': int(config['safe.listwise.validation.interval'])
        if config.contains('safe.listwise.validation.interval') else 2,
        'patience': int(config['safe.listwise.patience'])
        if config.contains('safe.listwise.patience') else 5,
        'min_delta': float(config['safe.listwise.min.delta'])
        if config.contains('safe.listwise.min.delta') else 1e-4,
        'recall_at_20_tolerance': float(
            config['safe.listwise.recall.at.20.tolerance']
        ) if config.contains(
            'safe.listwise.recall.at.20.tolerance'
        ) else 0.0,
    }
    if settings['base_expert'] not in {'prototype', 'jaccard'}:
        raise ValueError(
            'safe.listwise.base.expert must be prototype or jaccard.'
        )
    positive_keys = (
        'tau', 'hidden_size', 'gate_hidden_size', 'queries_per_update',
        'interval', 'temperature', 'validation_interval', 'patience',
    )
    for key in positive_keys:
        if settings[key] <= 0:
            raise ValueError('safe.listwise.%s must be positive.' % key)
    if settings['hard_unlabeled'] < 0 or settings['random_unlabeled'] < 0:
        raise ValueError('Safe listwise candidate counts cannot be negative.')
    if settings['hard_unlabeled'] + settings['random_unlabeled'] <= 0:
        raise ValueError('Safe listwise training requires unlabeled candidates.')
    for key in (
            'listwise_weight', 'pairwise_weight', 'bce_weight', 'trust_weight'):
        if settings[key] < 0:
            raise ValueError('safe.listwise.%s cannot be negative.' % key)
    if settings['listwise_weight'] <= 0:
        raise ValueError('safe.listwise.listwise_weight must be positive.')
    if settings['recall_at_20_tolerance'] < 0:
        raise ValueError(
            'safe.listwise.recall.at.20.tolerance cannot be negative.'
        )
    if enabled:
        transfer = herb_prototype_transfer or resolve_herb_prototype_transfer(
            config
        )
        if not transfer['enabled'] or not transfer['dual_granularity']:
            raise ValueError(
                'Safe listwise reranking requires fold-local prototype and '
                'Jaccard transfer states.'
            )
        distillation = (
            hc_ranking_distillation
            or resolve_hc_ranking_distillation(config, transfer)
        )
        if distillation['enabled']:
            raise ValueError(
                'Safe listwise reranking and H-C ranking distillation cannot '
                'be enabled together.'
            )
    return settings


class EarlyStoppingTracker(object):
    def __init__(self, patience, min_delta):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_value = None
        self.best_epoch = None
        self.stale_checks = 0

    def update(self, value, epoch):
        value = float(value)
        improved = self.best_value is None or value > self.best_value + self.min_delta
        if improved:
            self.best_value = value
            self.best_epoch = int(epoch)
            self.stale_checks = 0
        else:
            self.stale_checks += 1
        return improved, self.stale_checks >= self.patience


def build_regularization_loss(
        tf_module,
        weights,
        compound_embeddings,
        protein_embeddings,
        weight_decay,
        compound_regularization,
        protein_regularization):
    terms = [
        float(weight_decay) * tf_module.nn.l2_loss(weight)
        for weight in weights.values()
    ]
    terms.extend([
        float(compound_regularization) * tf_module.nn.l2_loss(compound_embeddings),
        float(protein_regularization) * tf_module.nn.l2_loss(protein_embeddings),
    ])
    return tf_module.add_n(terms)


def context_interaction_scores(
        compound_embeddings,
        protein_embeddings,
        compound_contexts,
        protein_contexts,
        compound_disease_weight,
        herb_protein_weight,
        herb_disease_weight,
        enabled_terms=None):
    compound_embeddings = np.asarray(compound_embeddings)
    protein_embeddings = np.asarray(protein_embeddings)
    compound_contexts = np.asarray(compound_contexts)
    protein_contexts = np.asarray(protein_contexts)

    if enabled_terms is None:
        enabled_terms = {
            'compound_disease': True,
            'herb_protein': True,
            'herb_disease': True,
        }
    scores = compound_embeddings.dot(protein_embeddings.transpose())
    if enabled_terms.get('compound_disease', False):
        scores += (compound_embeddings * compound_disease_weight).dot(protein_contexts.transpose())
    if enabled_terms.get('herb_protein', False):
        scores += (compound_contexts * herb_protein_weight).dot(protein_embeddings.transpose())
    if enabled_terms.get('herb_disease', False):
        scores += (compound_contexts * herb_disease_weight).dot(protein_contexts.transpose())
    return scores


def context_interaction_pair_scores(
        compound_embeddings,
        protein_embeddings,
        compound_contexts,
        protein_contexts,
        compound_indices,
        protein_indices,
        compound_disease_weight,
        herb_protein_weight,
        herb_disease_weight,
        enabled_terms=None,
        decoder_type='dot',
        decoder_weights=None,
        pair_compound_contexts=None,
        residual_compound_contexts=None,
        target_residual_weight=None,
        herb_protein_scale=None,
        base_score_scale=None,
        cold_herb_protein_weight=None,
        cold_herb_protein_scale=None,
        herb_disease_scale=None,
        herb_prototype_residual=None,
        herb_prototype_scale=None,
        herb_jaccard_residual=None,
        herb_jaccard_scale=None):
    compound_indices = np.asarray(compound_indices, dtype=np.int64)
    protein_indices = np.asarray(protein_indices, dtype=np.int64)
    if compound_indices.shape != protein_indices.shape:
        raise ValueError('Compound and protein index arrays must have the same shape.')

    compounds = np.asarray(compound_embeddings)[compound_indices]
    proteins = np.asarray(protein_embeddings)[protein_indices]
    static_herb_contexts = np.asarray(compound_contexts)[compound_indices]
    herb_contexts = (
        np.asarray(pair_compound_contexts)
        if pair_compound_contexts is not None
        else static_herb_contexts
    )
    if herb_contexts.shape != compounds.shape:
        raise ValueError('Pair compound contexts must match the pair embedding shape.')
    disease_contexts = np.asarray(protein_contexts)[protein_indices]
    if enabled_terms is None:
        enabled_terms = {
            'compound_disease': True,
            'herb_protein': True,
            'herb_disease': True,
        }

    scores = pair_decoder_scores(
        compounds, proteins, decoder_type=decoder_type, decoder_weights=decoder_weights
    )
    if base_score_scale is not None:
        base_score_scale = np.asarray(
            base_score_scale, dtype=np.float64
        ).reshape(-1)
        if base_score_scale.shape != scores.shape:
            raise ValueError(
                'Base-score scales must match the number of pairs.'
            )
        scores *= base_score_scale
    if enabled_terms.get('compound_disease', False):
        scores += np.sum(compounds * disease_contexts * compound_disease_weight, axis=1)
    if enabled_terms.get('herb_protein', False):
        herb_protein_scores = np.sum(
            herb_contexts * proteins * herb_protein_weight, axis=1
        )
        if herb_protein_scale is not None:
            herb_protein_scale = np.asarray(
                herb_protein_scale, dtype=np.float64
            ).reshape(-1)
            if herb_protein_scale.shape != herb_protein_scores.shape:
                raise ValueError(
                    'Herb-protein scales must match the number of pairs.'
                )
            herb_protein_scores *= herb_protein_scale
        scores += herb_protein_scores
        if residual_compound_contexts is not None:
            residual_contexts = np.asarray(residual_compound_contexts)
            if residual_contexts.shape != compounds.shape:
                raise ValueError(
                    'Residual compound contexts must match the pair embedding shape.'
                )
            if target_residual_weight is None:
                raise ValueError(
                    'Target residual weight is required with residual contexts.'
                )
            context_delta = residual_contexts - static_herb_contexts
            scores += np.sum(
                context_delta * proteins * np.asarray(target_residual_weight), axis=1
            )
        if cold_herb_protein_weight is not None:
            cold_herb_protein_scores = np.sum(
                static_herb_contexts
                * proteins
                * np.asarray(cold_herb_protein_weight),
                axis=1,
            )
            if cold_herb_protein_scale is not None:
                cold_herb_protein_scale = np.asarray(
                    cold_herb_protein_scale, dtype=np.float64
                ).reshape(-1)
                if cold_herb_protein_scale.shape != cold_herb_protein_scores.shape:
                    raise ValueError(
                        'Cold-expert scales must match the number of pairs.'
                    )
                cold_herb_protein_scores *= cold_herb_protein_scale
            scores += cold_herb_protein_scores
    if enabled_terms.get('herb_disease', False):
        herb_disease_scores = np.sum(
            herb_contexts * disease_contexts * herb_disease_weight,
            axis=1,
        )
        if herb_disease_scale is not None:
            herb_disease_scale = np.asarray(
                herb_disease_scale, dtype=np.float64
            ).reshape(-1)
            if herb_disease_scale.shape != herb_disease_scores.shape:
                raise ValueError(
                    'Herb-disease scales must match the number of pairs.'
                )
            herb_disease_scores *= herb_disease_scale
        scores += herb_disease_scores
    if herb_prototype_residual is not None:
        if herb_prototype_scale is None:
            raise ValueError(
                'Herb prototype scale is required with prototype residuals.'
            )
        prototype_residual = np.asarray(
            herb_prototype_residual, dtype=np.float64
        ).reshape(-1)
        if prototype_residual.shape != scores.shape:
            raise ValueError(
                'Herb prototype residuals must match the number of pairs.'
            )
        prototype_scale = float(
            np.asarray(herb_prototype_scale).reshape(-1)[0]
        )
        scores += prototype_scale * prototype_residual
    if herb_jaccard_residual is not None:
        if herb_jaccard_scale is None:
            raise ValueError(
                'Herb Jaccard scale is required with Jaccard residuals.'
            )
        jaccard_residual = np.asarray(
            herb_jaccard_residual, dtype=np.float64
        ).reshape(-1)
        if jaccard_residual.shape != scores.shape:
            raise ValueError(
                'Herb Jaccard residuals must match the number of pairs.'
            )
        jaccard_scale = float(
            np.asarray(herb_jaccard_scale).reshape(-1)[0]
        )
        scores += jaccard_scale * jaccard_residual
    return scores


def context_masked_pair_scores(
        compound_embeddings,
        protein_embeddings,
        compound_contexts,
        protein_contexts,
        compound_indices,
        protein_indices,
        compound_disease_weight,
        herb_protein_weight,
        herb_disease_weight,
        mask_compound=False,
        mask_protein=False,
        enabled_terms=None,
        decoder_type='dot',
        decoder_weights=None):
    compound_indices = np.asarray(compound_indices, dtype=np.int64)
    protein_indices = np.asarray(protein_indices, dtype=np.int64)
    if compound_indices.shape != protein_indices.shape:
        raise ValueError('Compound and protein index arrays must have the same shape.')
    compounds = np.asarray(compound_embeddings)[compound_indices]
    proteins = np.asarray(protein_embeddings)[protein_indices]
    herb_contexts = np.asarray(compound_contexts)[compound_indices]
    disease_contexts = np.asarray(protein_contexts)[protein_indices]
    effective_compounds = herb_contexts if mask_compound else compounds
    effective_proteins = disease_contexts if mask_protein else proteins
    if enabled_terms is None:
        enabled_terms = {
            'compound_disease': True,
            'herb_protein': True,
            'herb_disease': True,
        }
    scores = pair_decoder_scores(
        effective_compounds,
        effective_proteins,
        decoder_type=decoder_type,
        decoder_weights=decoder_weights,
    )
    if enabled_terms.get('compound_disease', False):
        scores += np.sum(
            effective_compounds * disease_contexts * compound_disease_weight,
            axis=1,
        )
    if enabled_terms.get('herb_protein', False):
        scores += np.sum(
            herb_contexts * effective_proteins * herb_protein_weight,
            axis=1,
        )
    if enabled_terms.get('herb_disease', False):
        scores += np.sum(
            herb_contexts * disease_contexts * herb_disease_weight,
            axis=1,
        )
    return scores


def target_conditioned_herb_contexts(
        herb_edge_embeddings,
        compound_herb_indices,
        compound_herb_mask,
        protein_embeddings,
        compound_indices,
        protein_indices,
        herb_projection,
        protein_projection,
        temperature=1.0):
    herb_edges = np.asarray(herb_edge_embeddings)
    incidence = np.asarray(compound_herb_indices, dtype=np.int64)
    incidence_mask = np.asarray(compound_herb_mask, dtype=np.float32)
    proteins = np.asarray(protein_embeddings)
    compound_indices = np.asarray(compound_indices, dtype=np.int64)
    protein_indices = np.asarray(protein_indices, dtype=np.int64)
    herb_projection = np.asarray(herb_projection)
    protein_projection = np.asarray(protein_projection)
    temperature = float(temperature)

    if temperature <= 0:
        raise ValueError('Target-attention temperature must be positive.')
    if herb_edges.ndim != 2 or proteins.ndim != 2:
        raise ValueError('Herb-edge and protein embeddings must be rank-2 arrays.')
    if incidence.shape != incidence_mask.shape or incidence.ndim != 2:
        raise ValueError('Compound-herb indices and mask must be matching rank-2 arrays.')
    if compound_indices.shape != protein_indices.shape:
        raise ValueError('Compound and protein index arrays must have the same shape.')
    dimension = herb_edges.shape[1]
    if proteins.shape[1] != dimension:
        raise ValueError('Herb-edge and protein embedding dimensions must match.')
    if herb_projection.shape != (dimension, dimension):
        raise ValueError('Herb projection must have shape [dimension, dimension].')
    if protein_projection.shape != (dimension, dimension):
        raise ValueError('Protein projection must have shape [dimension, dimension].')

    sentinel_index = herb_edges.shape[0]
    if np.any(incidence < 0) or np.any(incidence > sentinel_index):
        raise ValueError('Compound-herb incidence contains an invalid herb index.')
    padded_herb_edges = np.concatenate(
        [herb_edges, np.zeros((1, dimension), dtype=herb_edges.dtype)], axis=0
    )
    pair_incidence = incidence[compound_indices]
    pair_mask = incidence_mask[compound_indices]
    pair_herb_edges = padded_herb_edges[pair_incidence]
    herb_keys = np.matmul(pair_herb_edges, herb_projection)
    protein_queries = np.matmul(proteins[protein_indices], protein_projection)
    logits = np.sum(herb_keys * protein_queries[:, None, :], axis=2)
    logits = logits / (np.sqrt(float(dimension)) * temperature)
    masked_logits = np.where(pair_mask > 0, logits, -1e9)
    shifted_logits = masked_logits - np.max(masked_logits, axis=1, keepdims=True)
    unnormalized = np.exp(shifted_logits) * pair_mask
    normalizer = np.sum(unnormalized, axis=1, keepdims=True)
    attention = np.divide(
        unnormalized,
        normalizer,
        out=np.zeros_like(unnormalized),
        where=normalizer > 0,
    )
    contexts = np.sum(attention[:, :, None] * pair_herb_edges, axis=1)
    norms = np.linalg.norm(contexts, axis=1, keepdims=True)
    contexts = np.divide(
        contexts,
        norms,
        out=np.zeros_like(contexts),
        where=norms > 0,
    )
    return contexts, attention


def pair_decoder_scores(
        compound_vectors,
        protein_vectors,
        decoder_type='dot',
        decoder_weights=None):
    compounds = np.asarray(compound_vectors)
    proteins = np.asarray(protein_vectors)
    if compounds.shape != proteins.shape:
        raise ValueError('Compound and protein vectors must have the same shape.')
    decoder_type = str(decoder_type).strip().lower()
    decoder_weights = decoder_weights or {}

    dot_scores = np.sum(compounds * proteins, axis=1)
    if decoder_type == 'dot':
        return dot_scores
    if decoder_type == 'bilinear':
        if 'decoder_bilinear' not in decoder_weights:
            raise ValueError('Bilinear decoder weight is missing.')
        return np.sum(
            compounds.dot(np.asarray(decoder_weights['decoder_bilinear'])) * proteins,
            axis=1,
        )
    if decoder_type == 'mlp':
        required = (
            'decoder_mlp_hidden', 'decoder_mlp_hidden_bias',
            'decoder_mlp_output', 'decoder_mlp_output_bias',
        )
        missing = [key for key in required if key not in decoder_weights]
        if missing:
            raise ValueError('MLP decoder weights are missing: %s.' % ', '.join(missing))
        features = np.concatenate(
            [compounds, proteins, compounds * proteins, np.abs(compounds - proteins)],
            axis=1,
        )
        hidden = features.dot(np.asarray(decoder_weights['decoder_mlp_hidden']))
        hidden += np.asarray(decoder_weights['decoder_mlp_hidden_bias'])
        hidden = np.where(hidden >= 0, hidden, 0.2 * hidden)
        residual = hidden.dot(np.asarray(decoder_weights['decoder_mlp_output']))
        residual += np.asarray(decoder_weights['decoder_mlp_output_bias'])
        return dot_scores + np.reshape(residual, (-1,))
    raise ValueError('Unsupported pair decoder: %s.' % decoder_type)
