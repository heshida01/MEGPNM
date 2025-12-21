# -*- coding: utf-8 -*-
"""
Metrics utilities.

Includes common regression evaluation metrics.
"""
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr


def concordance_index(y_true, y_pred):
    """Compute Concordance Index (CI) using a vectorized implementation."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    n = len(y_true)
    if n < 2:
        return 0.0
    
    # Vectorized pairwise differences via broadcasting
    true_diff = y_true[:, np.newaxis] - y_true[np.newaxis, :]  # (n, n)
    pred_diff = y_pred[:, np.newaxis] - y_pred[np.newaxis, :]  # (n, n)
    
    # Only consider upper-triangular pairs (i < j)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # Exclude ties in y_true
    valid_pairs = mask & (true_diff != 0)
    
    if not valid_pairs.any():
        return 0.0
    
    # Concordant pairs: same sign
    concordant = ((true_diff > 0) & (pred_diff > 0) & valid_pairs) | \
                 ((true_diff < 0) & (pred_diff < 0) & valid_pairs)
    
    return concordant.sum() / valid_pairs.sum()


def adjusted_r2(y_true, y_pred, n_features):
    """Compute adjusted R²."""
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return adjusted_r2


def pearson_correlation(y_true, y_pred):
    """Compute Pearson correlation coefficient (PCC)."""
    try:
        # Need enough samples and non-zero variance
        if len(y_true) < 2:
            return np.nan
        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            return np.nan
        correlation, p_value = pearsonr(y_true, y_pred)
        return correlation
    except Exception as e:
        # Return NaN rather than a misleading 0
        print(f"Warning: Failed to compute Pearson correlation: {e}")
        return np.nan


def spearman_correlation(y_true, y_pred):
    """Compute Spearman correlation coefficient (SCC)."""
    try:
        # Need enough samples
        if len(y_true) < 2:
            return np.nan
        # Spearman fails for constant vectors
        if len(np.unique(y_true)) == 1 or len(np.unique(y_pred)) == 1:
            return np.nan
        correlation, p_value = spearmanr(y_true, y_pred)
        return correlation
    except Exception as e:
        # Return NaN rather than a misleading 0
        print(f"Warning: Failed to compute Spearman correlation: {e}")
        return np.nan


def calculate_metrics(y_true, y_pred, n_features=None):
    """
    Calculate all supported metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        n_features: Number of features (for adjusted R²).

    Returns:
        A dict of metrics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    results = {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': float(r2_score(y_true, y_pred)),
        'ci': float(concordance_index(y_true, y_pred)),
        'pcc': float(pearson_correlation(y_true, y_pred)),
        'scc': float(spearman_correlation(y_true, y_pred))
    }
    
    if n_features is not None:
        results['adj_r2'] = float(adjusted_r2(y_true, y_pred, n_features))
    
    return results
