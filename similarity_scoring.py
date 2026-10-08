"""A bounded ranking score, not a calibrated probability."""

import numpy as np

SCORE_DESCRIPTION = 'Distance score: clip(1 - Euclidean distance / 2, 0, 1); not a probability'


def distance_scores(distances):
    distances = np.asarray(distances, dtype=float)
    if not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError('Distances must be finite and non-negative')
    return np.clip(1.0 - distances / 2.0, 0.0, 1.0)
