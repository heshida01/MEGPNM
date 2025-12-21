# -*- coding: utf-8 -*-
"""
Explainability utilities.

Provides Grad-CAM and related analysis helpers.
"""
import os
import copy
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import dgl
from .visualization import visualize_gradcam_heatmap


def get_node_importance_scores(model, graph, method='gap', device='cuda'):
    """
    Get node importance scores.

    Args:
        model: Trained model.
        graph: DGL graph.
        method: 'gap' or 'grad'.
        device: Device.
    Returns:
        node_scores: numpy array of node importance scores.
    """
    model.eval()
    graph = graph.to(device)
    
    if method == 'gap':
        # Use GlobalAttentionPooling attention weights
        with torch.no_grad():
            output, node_attention, _ = model(graph, return_attention=True)
            if node_attention is not None:
                node_scores = node_attention.squeeze().cpu().numpy()
                # Ensure correct shape
                if node_scores.ndim == 0:
                    node_scores = np.array([node_scores])
            else:
                # If attention weights are unavailable, use a uniform distribution
                node_scores = np.ones(graph.num_nodes()) / graph.num_nodes()
    
    elif method == 'grad':
        # Gradient × Input
        graph.ndata['feat'].requires_grad_(True)
        output = model(graph)
        
        # Gradient of output w.r.t. node features
        grad_outputs = torch.ones_like(output)
        gradients = torch.autograd.grad(
            outputs=output,
            inputs=graph.ndata['feat'],
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False,
            only_inputs=True
        )[0]
        
        # Sum |gradient × input| across feature dims
        node_scores = (torch.abs(gradients * graph.ndata['feat'])).sum(dim=1).detach().cpu().numpy()
        
        # Clear gradients
        graph.ndata['feat'].requires_grad_(False)
        
    else:
        raise ValueError(f"Unsupported importance method: {method}")
    
    return node_scores


def mask_graph_nodes(graph, node_indices, mask_mode='zero_feature'):
    """
    Mask specified nodes in a graph.

    Args:
        graph: DGL graph.
        node_indices: Node indices to mask.
        mask_mode: 'zero_feature' or 'remove_node'.
    Returns:
        masked_graph: Masked graph.
    """
    if mask_mode == 'zero_feature':
        # Zero-out features for selected nodes
        masked_graph = copy.deepcopy(graph)
        if len(node_indices) > 0:
            # Keep indices in range
            valid_indices = [idx for idx in node_indices if idx < masked_graph.num_nodes()]
            if valid_indices:
                masked_graph.ndata['feat'][valid_indices] = 0.0
        return masked_graph
    
    elif mask_mode == 'remove_node':
        # Removing nodes requires rebuilding graph structure; fall back to feature masking.
        print("Warning: remove_node mode not fully implemented, using zero_feature instead")
        return mask_graph_nodes(graph, node_indices, 'zero_feature')
    
    else:
        raise ValueError(f"Unsupported mask mode: {mask_mode}")


# ========== Grad-CAM ==========

def compute_gradcam_scores(activations, gradients, apply_relu=True, norm_method='minmax'):
    """
    Compute Grad-CAM scores.

    Args:
        activations: Activations [N, D].
        gradients: Gradients [N, D].
        apply_relu: Whether to apply ReLU.
        norm_method: Normalization method.
    Returns:
        scores: Node importance scores [N].
    """
    if activations is None or gradients is None:
        return None
    
    # Ensure torch tensors
    if not isinstance(activations, torch.Tensor):
        activations = torch.tensor(activations)
    if not isinstance(gradients, torch.Tensor):
        gradients = torch.tensor(gradients)
    
    # Global average gradient per channel (channel importance weights)
    alpha = gradients.mean(dim=0, keepdim=True)  # [1, D]
    
    # Node heat score: weighted sum of activations
    scores = (activations * alpha).sum(dim=1)  # [N]
    
    # Optional ReLU (keep positive contributions)
    if apply_relu:
        scores = torch.relu(scores)
    
    # Convert to numpy
    scores = scores.detach().cpu().numpy()
    
    # Normalize
    if norm_method == 'minmax' and len(scores) > 0:
        score_min = scores.min()
        score_max = scores.max()
        if score_max != score_min:
            scores = (scores - score_min) / (score_max - score_min)
        else:
            # If all scores are identical, set to 0.5
            scores = np.ones_like(scores) * 0.5
    
    return scores


def extract_gradcam_for_batch(model, batch_graph, batch_labels, hook, target_layer, 
                             apply_relu=True, norm_method='minmax', device='cuda'):
    """
    Extract Grad-CAM scores for a batched graph (optimized autograd.grad version).

    Args:
        model: Model.
        batch_graph: Batched DGL graph.
        batch_labels: Batched labels.
        hook: GradCAM hook object (unused).
        target_layer: Target layer.
        apply_relu: Whether to apply ReLU.
        norm_method: Normalization method.
        device: Device.
    Returns:
        scores_list: List of per-molecule score arrays.
        predictions: Predictions.
    """
    model.eval()
    batch_graph = batch_graph.to(device)
    
    # Forward once to get predictions and target activations
    predictions = model(batch_graph)  # [B, 1] or [B]
    
    # Target activations
    if target_layer == 'pre_pool':
        if hasattr(model, 'pre_pool_output'):
            target_activations = model.pre_pool_output  # [N_total, D]
        else:
            raise ValueError('Model does not have pre_pool_output attribute')
    elif target_layer == 'last_gat':
        if hasattr(model, 'last_gat_output'):
            target_activations = model.last_gat_output  # [N_total, D]
        else:
            raise ValueError('Model does not have last_gat_output attribute')
    else:
        raise ValueError(f'Unknown target_layer: {target_layer}')
    
    # Per-sample gradients (memory-safe: one VJP per sample)
    scores_list = []
    node_counts = batch_graph.batch_num_nodes().tolist()
    start = 0
    B = len(node_counts)
    
    for b, n in enumerate(node_counts):
        end = start + n
        model.zero_grad(set_to_none=True)
        
        try:
            # Ensure scalar output; provide explicit grad_outputs when needed
            out_b = predictions[b]
            if out_b.numel() == 1:
                grads = torch.autograd.grad(
                    outputs=out_b,
                    inputs=target_activations,
                    grad_outputs=torch.ones_like(out_b),
                    retain_graph=(b < B - 1),
                    allow_unused=False
                )[0]
            else:
                grads = torch.autograd.grad(
                    outputs=out_b.sum(),
                    inputs=target_activations,
                    retain_graph=(b < B - 1),
                    allow_unused=False
                )[0]
            
            # Slice node range for this sample
            act_b = target_activations[start:end]   # [n, D]
            grad_b = grads[start:end]                # [n, D]
            
            # Grad-CAM: channel-wise mean gradient -> channel weights -> node heat
            scores = compute_gradcam_scores(
                activations=act_b.detach(), 
                gradients=grad_b.detach(), 
                apply_relu=apply_relu, 
                norm_method=norm_method
            )
            
            if scores is None or len(scores) == 0:
                scores = np.ones(n, dtype=np.float32) / max(n, 1)
                
        except Exception as e:
            print(" Sample {} gradient computation failed: {}".format(b, str(e)))
            scores = np.ones(n, dtype=np.float32) / max(n, 1)
        
        scores_list.append(scores)
        start = end
    
    return scores_list, predictions.detach().cpu().numpy()


def select_samples_for_analysis(test_dataset, num_samples, sort_by='high_true_value', seed=42):
    """
    Select samples for analysis.

    Args:
        test_dataset: Test dataset.
        num_samples: Number of samples to select.
        sort_by: Sorting strategy.
        seed: Random seed.
    Returns:
        selected_indices: Selected indices.
        sample_info: Optional sample metadata.
    """
    total_samples = len(test_dataset)
    if num_samples <= 0 or num_samples >= total_samples:
        return list(range(total_samples)), None
    
    # Collect sample metadata for sorting
    sample_data = []
    for i in range(total_samples):
        item = test_dataset[i]
        true_value = item['label'].item()
        sample_data.append({
            'index': i,
            'true_value': true_value,
            'smiles': item['smiles']
        })
    
    # Select according to strategy
    if sort_by == 'high_true_value':
        sample_data.sort(key=lambda x: x['true_value'], reverse=True)
    elif sort_by == 'low_true_value':
        sample_data.sort(key=lambda x: x['true_value'], reverse=False)
    elif sort_by == 'random':
        np.random.seed(seed)
        np.random.shuffle(sample_data)
    else:
        # Other strategies may require model predictions; fall back to random
        np.random.seed(seed)
        np.random.shuffle(sample_data)
    
    selected_indices = [item['index'] for item in sample_data[:num_samples]]
    sample_info = sample_data[:num_samples]
    
    return selected_indices, sample_info


def run_gradcam_analysis(model, test_dataset, device='cuda', target_layer='pre_pool',
                        num_samples=50, sort_by='high_true_value', output_dir='gradcam_analysis',
                        apply_relu=True, norm_method='minmax', overlay_alpha=0.7):
    """
    Run a full Grad-CAM analysis.
    """
    print("\n{}".format('='*70))
    print(" Starting Grad-CAM analysis")
    print("{}".format('='*70))
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Select samples
    print(" Selecting samples...")
    selected_indices, sample_info = select_samples_for_analysis(
        test_dataset, num_samples, sort_by, seed=42
    )
    
    print("Target layer: {}".format(target_layer))
    print("Sampling strategy: {}".format(sort_by))
    print("Selected samples: {}".format(len(selected_indices)))
    print("ReLU: {}".format('yes' if apply_relu else 'no'))
    print("Normalization: {}".format(norm_method))
    
    if sample_info:
        true_values = [item['true_value'] for item in sample_info]
        print("True value range: {:.3f} ~ {:.3f}".format(min(true_values), max(true_values)))
    
    # Batch samples
    batch_size = 8  # small batches to save memory
    results = {
        'analysis_type': 'gradcam_molecular_heatmap',
        'timestamp': pd.Timestamp.now().isoformat(),
        'parameters': {
            'target_layer': target_layer,
            'num_samples': len(selected_indices),
            'sort_by': sort_by,
            'apply_relu': apply_relu,
            'norm_method': norm_method,
            'overlay_alpha': overlay_alpha
        },
        'sample_details': [],
        'statistics': {
            'total_samples': len(selected_indices),
            'successful_visualizations': 0,
            'failed_visualizations': 0,
            'avg_score_range': 0.0,
            'avg_top3_score': 0.0
        }
    }
    
    score_ranges = []
    top3_scores = []
    
    print("\n Extracting Grad-CAM heatmaps...")
    
    for batch_start in tqdm(range(0, len(selected_indices), batch_size), desc="Batches"):
        batch_end = min(batch_start + batch_size, len(selected_indices))
        batch_indices = selected_indices[batch_start:batch_end]
        
        # Gather batch data
        batch_graphs = []
        batch_labels = []
        batch_smiles = []
        batch_true_values = []
        
        for idx in batch_indices:
            item = test_dataset[idx]
            batch_graphs.append(item['graph'])
            batch_labels.append(item['label'])
            batch_smiles.append(item['smiles'])
            batch_true_values.append(item['label'].item())
        
        # Create DGL batch
        batched_graph = dgl.batch(batch_graphs)
        batched_labels = torch.stack(batch_labels)
        
        try:
            # Extract Grad-CAM scores
            scores_list, predictions = extract_gradcam_for_batch(
                model, batched_graph, batched_labels, None, target_layer,
                apply_relu, norm_method, device
            )
            
            # Per-sample post-processing
            for i, (idx, scores, pred, true_val, smiles) in enumerate(
                zip(batch_indices, scores_list, predictions, batch_true_values, batch_smiles)):
                
                if len(scores) == 0:
                    results['statistics']['failed_visualizations'] += 1
                    continue
                
                # Generate visualization
                viz_filename = f'gradcam_{batch_start+i+1:03d}_{idx}_{sort_by}.png'
                viz_path = os.path.join(output_dir, viz_filename)
                
                success = visualize_gradcam_heatmap(
                    smiles, scores, viz_path, overlay_alpha
                )
                
                if success:
                    results['statistics']['successful_visualizations'] += 1
                    
                    # Stats
                    score_range = scores.max() - scores.min()
                    top3_indices = np.argsort(scores)[-3:]
                    top3_score = np.mean(scores[top3_indices])
                    
                    score_ranges.append(score_range)
                    top3_scores.append(top3_score)
                    
                    # Save sample details
                    sample_detail = {
                        'sample_index': int(idx),
                        'smiles': smiles,
                        'true_value': float(true_val),
                        'predicted_value': float(pred),
                        'score_range': float(score_range),
                        'top3_score': float(top3_score),
                        'top3_atoms': top3_indices.tolist(),
                        'all_scores': scores.tolist(),
                        'visualization_path': viz_filename
                    }
                    results['sample_details'].append(sample_detail)
                    
                else:
                    results['statistics']['failed_visualizations'] += 1
        
        except Exception as e:
            print(" Batch {}-{} failed: {}".format(batch_start, batch_end, str(e)))
            results['statistics']['failed_visualizations'] += batch_end - batch_start
    
    # Aggregate statistics
    if score_ranges:
        results['statistics']['avg_score_range'] = float(np.mean(score_ranges))
        results['statistics']['avg_top3_score'] = float(np.mean(top3_scores))
    
    # Save results
    results_path = os.path.join(output_dir, 'gradcam_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save a CSV summary
    if results['sample_details']:
        csv_data = []
        for detail in results['sample_details']:
            csv_row = {
                'sample_index': detail['sample_index'],
                'smiles': detail['smiles'],
                'true_value': detail['true_value'],
                'predicted_value': detail['predicted_value'],
                'score_range': detail['score_range'],
                'top3_score': detail['top3_score'],
                'visualization_path': detail['visualization_path']
            }
            csv_data.append(csv_row)
        
        csv_path = os.path.join(output_dir, 'gradcam_summary.csv')
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    
    # Print summary
    print("\n Grad-CAM summary:")
    print("  - Target layer: {}".format(target_layer))
    print("  - Processed: {}".format(results['statistics']['total_samples']))
    print("  - Successful: {}".format(results['statistics']['successful_visualizations']))
    print("  - Failed: {}".format(results['statistics']['failed_visualizations']))
    if score_ranges:
        print("  - Avg score range: {:.4f}".format(results['statistics']['avg_score_range']))
        print("  - Avg Top3 score: {:.4f}".format(results['statistics']['avg_top3_score']))
    
    print("\n Grad-CAM analysis completed")
    print(" Output directory: {}{}".format(output_dir, "/"))
    print("  - gradcam_results.json: full results")
    print("  - gradcam_summary.csv: summary table")
    print("  - *.png: molecular heatmaps")
    
    return results
