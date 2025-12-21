# -*- coding: utf-8 -*-
"""
NpcGAT package.

Exports core modules used by `main.py`.
"""

from .utils import set_random_seed, ENHANCED_ATOM_FEATURE_DIM
from .features import (
    get_bond_features, 
    one_of_k_encoding, 
    one_of_k_encoding_unk, 
    get_atom_features, 
    smiles_to_dgl_graph, 
    validate_atom_features
)
from .dataset import GraphDataset, graph_collate_fn
from .models import JumpingKnowledge, EnhancedGATModel
from .metrics import concordance_index, adjusted_r2, pearson_correlation, spearman_correlation
from .visualization import (
    visualize_attention, 
    explain_subgraph, 
    visualize_molecule_attention, 
    visualize_gradcam_heatmap, 
    create_joint_analysis_plots
)
from .explainability import (
    get_node_importance_scores,
    mask_graph_nodes,
    compute_gradcam_scores,
    extract_gradcam_for_batch,
    run_gradcam_analysis,
    select_samples_for_analysis
)
from .training import train_model, evaluate_model

__all__ = [
    # utils
    'set_random_seed', 'ENHANCED_ATOM_FEATURE_DIM',
    # features
    'get_bond_features', 'one_of_k_encoding', 'one_of_k_encoding_unk', 
    'get_atom_features', 'smiles_to_dgl_graph', 'validate_atom_features',
    # dataset
    'GraphDataset', 'graph_collate_fn',
    # models
    'EdgeGATLayer', 'JumpingKnowledge', 'EnhancedGATModel', 'EdgeGCNLayer', 'EnhancedGCNModel',
    # metrics
    'concordance_index', 'adjusted_r2', 'pearson_correlation', 'spearman_correlation',
    # visualization
    'visualize_attention', 'explain_subgraph', 'visualize_molecule_attention', 
    'visualize_gradcam_heatmap', 'create_joint_analysis_plots',
    # explainability
    'get_node_importance_scores', 'mask_graph_nodes',
    'compute_gradcam_scores', 'extract_gradcam_for_batch',
    'run_gradcam_analysis', 'select_samples_for_analysis',
    # training
    'train_model', 'evaluate_model'
]
