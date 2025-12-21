# -*- coding: utf-8 -*-
"""
Visualization utilities.

Includes molecule-level attention visualization and Grad-CAM plotting helpers.
"""
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from .features import smiles_to_dgl_graph


def visualize_attention(smiles, node_scores=None, edge_scores=None, graph=None, save_path=None):
    """Enhanced visualization supporting node/edge overlays."""
    try:
        from enhanced_visualization import AttentionAnalyzer
        analyzer = AttentionAnalyzer()
        return analyzer.create_attention_heatmap(
            smiles=smiles, 
            node_scores=node_scores, 
            edge_scores=edge_scores,
            graph=graph,
            save_path=save_path
        )
    except Exception as e:
        print(" Enhanced visualization failed; falling back to a simple implementation: {}{}".format(str(e), ""))
        # Fallback implementation
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Simple highlighting
        highlight_atoms = []
        if node_scores is not None:
            norm_scores = (node_scores - node_scores.min()) / (node_scores.max() - node_scores.min() + 1e-6)
            threshold = np.percentile(norm_scores, 70)
            highlight_atoms = [i for i, score in enumerate(norm_scores) if score > threshold and i < mol.GetNumAtoms()]
        
        # Simple RDKit rendering
        img = Draw.MolToImage(mol, size=(400, 300), highlightAtoms=highlight_atoms)
        
        if save_path:
            img.save(save_path)
        
        return img


def explain_subgraph(model, smiles, device='cuda', top_k=5):
    """Explain a molecule subgraph (node and edge importance)."""
    try:
        from enhanced_visualization import AttentionAnalyzer
        analyzer = AttentionAnalyzer(model)
        graph = smiles_to_dgl_graph(smiles, model.use_edge_feat).to(device)
        attention_data = analyzer.extract_multilayer_attention(graph, smiles)
        
        node_scores = None
        edge_scores = None
        
        # Node scores
        if attention_data['global_attention'] is not None:
            node_scores = attention_data['global_attention'].squeeze()
        
        # Edge scores (average edge attention across layers)
        if attention_data['edge_attention']:
            all_edge_attentions = []
            for layer_name, edge_attn in attention_data['edge_attention'].items():
                if edge_attn is not None:
                    # Average heads if needed
                    if edge_attn.ndim > 1:
                        avg_attn = np.mean(edge_attn, axis=1)
                    else:
                        avg_attn = edge_attn
                    all_edge_attentions.append(avg_attn)
            
            if all_edge_attentions:
                # Average across layers
                edge_scores = np.mean(all_edge_attentions, axis=0)
        
        # Top-k nodes
        top_indices = None
        if node_scores is not None:
            top_indices = np.argsort(node_scores)[-top_k:] if len(node_scores) > top_k else np.arange(len(node_scores))
        
        return {
            'top_node_indices': top_indices,
            'node_scores': node_scores,
            'edge_scores': edge_scores,
            'graph': graph
        }
    except Exception as e:
        print(" Subgraph explanation failed: {}{}".format(str(e), ""))
        return None


def visualize_molecule_attention(model, smiles, device='cuda', save_path=None):
    """
    Full molecule attention visualization.
    
    Args:
        model: Trained GNN model.
        smiles: SMILES string.
        device: Device.
        save_path: Output path.
        
    Returns:
        Rendered image (or None).
    """
    try:
        # Extract attention data
        explanation_data = explain_subgraph(model, smiles, device)
        if explanation_data is None:
            return None
        
        # Visualize
        img = visualize_attention(
            smiles=smiles,
            node_scores=explanation_data['node_scores'],
            edge_scores=explanation_data['edge_scores'],
            graph=explanation_data['graph'],
            save_path=save_path
        )
        
        return img
    except Exception as e:
        print(" Molecule attention visualization failed: {}{}".format(str(e), ""))
        return None


def visualize_gradcam_heatmap(smiles, scores, output_path, overlay_alpha=0.7, 
                             molecule_size=(400, 300)):
    """
    Generate a Grad-CAM heatmap visualization (simplified implementation).

    Args:
        smiles: SMILES string.
        scores: Atom importance scores.
        output_path: Output image path.
        overlay_alpha: Overlay alpha.
        molecule_size: Molecule image size.
    Returns:
        Whether the image was generated successfully.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(" Failed to parse SMILES: {}{}".format(smiles, ""))
            return False
        
        # Ensure score count matches atom count
        num_atoms = mol.GetNumAtoms()
        if len(scores) != num_atoms:
            print(" Atom count ({}) != score count ({}), SMILES: {}".format(num_atoms, len(scores), smiles))
            return False
        
        # Normalize scores to [0, 1]
        scores = np.array(scores)
        if scores.max() != scores.min():
            norm_scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            norm_scores = np.ones_like(scores) * 0.5  # constant
        
        # Top-k highlighting with a richer colormap
        try:
            import matplotlib.cm as cm
            cmap = cm.get_cmap('jet')
            
            # Color only the top-k atoms
            k = max(3, int(0.2 * num_atoms))
            topk_idx = np.argsort(norm_scores)[-k:]
            highlight_atoms = topk_idx.tolist()
            atom_colors = {}
            for i in highlight_atoms:
                r, g, b, _ = cmap(float(norm_scores[i]))
                atom_colors[i] = (r, g, b)
            
            # Use larger radii for highlight atoms
            highlight_radii = {i: 0.7 for i in highlight_atoms}
            
            # Render image
            img = Draw.MolToImage(
                mol,
                size=molecule_size,
                highlightAtoms=highlight_atoms,
                highlightAtomColors=atom_colors,
                highlightAtomRadii=highlight_radii,
                highlightBonds=[]
            )
            
            # Save image
            img.save(output_path)
            return True
            
        except Exception as e1:
            print(" Standard rendering failed: {}{}".format(str(e1), "; trying fallback"))
            
            # Fallback: plain drawing
            try:
                img = Draw.MolToImage(mol, size=molecule_size)
                img.save(output_path)
                return True
            except Exception as e2:
                print(" Fallback rendering also failed: {}{}".format(str(e2), ""))
                return False
        
    except Exception as e:
        print(" Grad-CAM visualization failed completely: {}{}".format(str(e), ""))
        return False


def create_joint_analysis_plots(joint_results, output_dir):
    """Create comparison plots for joint analysis outputs."""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1) Fidelity curves
        fidelity = joint_results['fidelity_results']
        ratios = np.array(fidelity['ratios'])
        deletion_curve = np.array(fidelity['pos_curve'])
        insertion_curve = np.array(fidelity['neg_curve'])
        
        ax1.plot(ratios, deletion_curve, 'r-o', label=f"Deletion (AUC: {fidelity['pos_auc']:.4f})", linewidth=2)
        ax1.plot(ratios, insertion_curve, 'b-s', label=f"Insertion (AUC: {fidelity['neg_auc']:.4f})", linewidth=2)
        ax1.set_xlabel('Mask Ratio')
        ax1.set_ylabel('Relative Change')
        ax1.set_title('Fidelity Analysis')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2) Attention entropy distribution
        explainability = joint_results['explainability_results']
        entropies = [s['attention_entropy'] for s in explainability['sample_details']]
        ax2.hist(entropies, bins=15, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(np.mean(entropies), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(entropies):.3f}')
        ax2.set_xlabel('Attention Entropy')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Attention Entropy Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3) Attention concentration distribution
        concentrations = [s['attention_concentration'] for s in explainability['sample_details']]
        ax3.hist(concentrations, bins=15, alpha=0.7, color='orange', edgecolor='black')
        ax3.axvline(np.mean(concentrations), color='red', linestyle='--',
                   label=f'Mean: {np.mean(concentrations):.3f}')
        ax3.set_xlabel('Attention Concentration')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Attention Concentration Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4) True value vs attention entropy
        true_values = [s['true_value'] for s in explainability['sample_details']]
        if len(true_values) > 0 and len(entropies) > 0:
            scatter = ax4.scatter(true_values, entropies, alpha=0.6, c=concentrations, cmap='viridis')
            ax4.set_xlabel('True Value')
            ax4.set_ylabel('Attention Entropy')
            ax4.set_title('True Value vs Attention Entropy')
            if len(concentrations) > 0:
                cbar = plt.colorbar(scatter, ax=ax4)
                cbar.set_label('Attention Concentration')
        else:
            ax4.text(0.5, 0.5, 'No data available', transform=ax4.transAxes, 
                    ha='center', va='center', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        import os
        plot_path = os.path.join(output_dir, 'joint_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(" Comparison plot saved: {}{}".format(plot_path, ""))
        
    except Exception as e:
        print(" Failed to generate comparison plot: {}{}".format(str(e), ""))
