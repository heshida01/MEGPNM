# -*- coding: utf-8 -*-
"""
General utilities.

Includes random seed setup, shared constants, and other common helpers.
"""
import os
import random
import numpy as np
import torch
import dgl
import warnings
import rdkit.rdBase as rkrb

# Suppress warnings/logs
warnings.filterwarnings('ignore')
rkrb.DisableLog('rdApp.error')
rkrb.DisableLog('rdApp.warning')

# Model input dimensionality (enhanced atom features)
ENHANCED_ATOM_FEATURE_DIM = 110  # enhanced atom feature dimension


def set_random_seed(seed=42):
    """Set random seeds for full reproducibility."""
    print("Setting random seed to: {}".format(seed))
    
    # Python built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # DGL
    dgl.random.seed(seed)
    
    # Prefer deterministic algorithms (warn_only keeps training usable)
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # cuDNN determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    
    # Environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Disable TF32 for numerical stability
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = False
    
    # Use a single CPU thread to reduce non-determinism
    torch.set_num_threads(1)
    
    print("Random seed setup complete, using deterministic algorithms")
