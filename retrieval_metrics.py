"""Retrieval metrics on a held-out gallery, with self matches excluded."""

import numpy as np


def retrieval_metrics(embeddings, labels, ks=(1, 5, 10), seed=42, bootstrap_samples=1000):
    embeddings = np.asarray(embeddings, dtype=float)
    labels = np.asarray(labels)
    if (embeddings.ndim != 2 or len(embeddings) != len(labels) or len(labels) < 2
            or not np.isfinite(embeddings).all()):
        raise ValueError('Expected finite embeddings with one label per image')
    if any(k < 1 for k in ks) or bootstrap_samples < 1:
        raise ValueError('k and bootstrap sample counts must be positive')
    counts = {str(label): int(np.sum(labels == label)) for label in np.unique(labels)}
    if len(counts) < 2 or min(counts.values()) < 2:
        raise ValueError('Evaluation needs two categories and a relevant peer for every query')
    values = {'average_precision': [], **{f'{metric}@{k}': [] for k in ks
                                         for metric in ('precision', 'recall', 'hit_rate')}}
    per_category = {}
    for i, query in enumerate(embeddings):
        distances = np.linalg.norm(embeddings - query, axis=1)
        candidates = np.delete(np.arange(len(labels)), i)
        ranked = candidates[np.argsort(distances[candidates], kind='stable')]
        relevant = labels[ranked] == labels[i]
        total_relevant = int(relevant.sum())
        ap = float(np.sum(np.cumsum(relevant) / np.arange(1, len(ranked) + 1) * relevant) / total_relevant)
        values['average_precision'].append(ap)
        per_category.setdefault(str(labels[i]), []).append(ap)
        for k in ks:
            effective_k = min(k, len(ranked))
            hits = int(relevant[:effective_k].sum())
            values[f'precision@{k}'].append(hits / effective_k)
            values[f'recall@{k}'].append(hits / total_relevant)
            values[f'hit_rate@{k}'].append(float(hits > 0))
    rng = np.random.default_rng(seed)
    metrics = {}
    for name, samples in values.items():
        samples = np.asarray(samples)
        means = [float(rng.choice(samples, size=len(samples), replace=True).mean())
                 for _ in range(bootstrap_samples)]
        metrics['mAP' if name == 'average_precision' else name] = {
            'value': float(samples.mean()),
            'query_bootstrap_95_ci': np.quantile(means, [0.025, 0.975]).tolist()}
    return {'query_count': len(labels), 'gallery_count': len(labels),
            'self_matches_excluded': True, 'category_counts': counts,
            'effective_k': {str(k): min(k, len(labels) - 1) for k in ks},
            'metrics': metrics,
            'per_category_mAP': {label: float(np.mean(aps)) for label, aps in per_category.items()},
            'macro_category_mAP': float(np.mean([np.mean(aps) for aps in per_category.values()])),
            'ci_note': 'Query bootstrap is descriptive: queries share a gallery and may be correlated. '
                       'It does not measure uncertainty across training seeds or datasets.'}
