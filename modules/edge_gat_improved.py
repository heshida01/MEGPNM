# -*- coding: utf-8 -*-
"""
Improved edge-aware GAT implementations.

Provides multiple strategies for fusing edge features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch import GATv2Conv


class EdgeGATv2Layer(nn.Module):
    """
    Edge-aware GATv2 layer.

    Strategy 1: inject edge features into the attention computation.
    """
    def __init__(self, in_dim, out_dim, edge_dim, num_heads=4, 
                 feat_drop=0.1, attn_drop=0.1, negative_slope=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        
        # Output dim per head
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        
        # Node feature transforms
        self.fc_src = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_dst = nn.Linear(in_dim, out_dim, bias=False)
        
        # Edge feature transform (project to the same space as node features)
        if edge_dim > 0:
            self.fc_edge = nn.Linear(edge_dim, out_dim, bias=False)
        
        # GATv2-style attention parameter
        # attention score = a^T * LeakyReLU(W_src*h_i + W_dst*h_j + W_edge*e_ij)
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_heads, self.head_dim)))
        
        # Dropout and activation
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_dim))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Parameter initialization."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        if hasattr(self, 'fc_edge'):
            nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.attn, gain=gain)
        
    def forward(self, graph, node_feat, edge_feat=None):
        """
        Forward pass.

        Returns:
            new_node_feat: Updated node features [N, out_dim]
            attention_weights: Attention weights [E, num_heads]
        """
        with graph.local_scope():
            N = node_feat.shape[0]
            
            # Apply dropout
            node_feat = self.feat_drop(node_feat)
            if edge_feat is not None:
                edge_feat = self.feat_drop(edge_feat)
            
            # Node feature transforms
            feat_src = self.fc_src(node_feat).view(N, self.num_heads, self.head_dim)
            feat_dst = self.fc_dst(node_feat).view(N, self.num_heads, self.head_dim)
            
            # Store on graph
            graph.ndata['ft_src'] = feat_src
            graph.ndata['ft_dst'] = feat_dst
            
            # Base attention (without edge features)
            graph.apply_edges(fn.u_add_v('ft_src', 'ft_dst', 'e_nodes'))
            e = graph.edata['e_nodes']  # [E, num_heads, head_dim]
            
            # Add edge feature contribution
            if edge_feat is not None and self.edge_dim > 0:
                edge_feat_transformed = self.fc_edge(edge_feat).view(-1, self.num_heads, self.head_dim)
                e = e + edge_feat_transformed
            
            # Attention scores
            e = self.leaky_relu(e)  # [E, num_heads, head_dim]
            e = (e * self.attn).sum(dim=-1).unsqueeze(-1)  # [E, num_heads, 1]
            
            # Softmax normalization
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            
            # Message passing
            graph.update_all(fn.u_mul_e('ft_src', 'a', 'm'),
                           fn.sum('m', 'ft'))
            
            rst = graph.ndata['ft']  # [N, num_heads, head_dim]
            
            # Reshape output and add bias
            rst = rst.reshape(N, self.out_dim) + self.bias
            
            # Return results and attention weights
            attention = graph.edata['a'].squeeze(-1)  # [E, num_heads]
            
            return rst, attention


class EdgeFusionGATLayer(nn.Module):
    """
    Edge-aware GAT layer.

    Strategy 2: fuse edge features via message passing.
    """
    def __init__(self, in_dim, out_dim, edge_dim, num_heads=4,
                 feat_drop=0.1, attn_drop=0.1):
        super().__init__()
        
        # Standard GATv2Conv for node features
        self.gat = GATv2Conv(in_dim, out_dim // num_heads, 
                            num_heads=num_heads,
                            feat_drop=feat_drop, 
                            attn_drop=attn_drop,
                            allow_zero_in_degree=True)
        
        # Edge feature encoder
        if edge_dim > 0:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ELU(),
                nn.Dropout(feat_drop)
            )
            
            # Fusion layer: combine node and aggregated edge information
            self.fusion = nn.Sequential(
                nn.Linear(out_dim * 2, out_dim),
                nn.LayerNorm(out_dim),
                nn.ELU(),
                nn.Dropout(feat_drop)
            )
        else:
            self.edge_encoder = None
            self.fusion = None
            
        self.num_heads = num_heads
        self.out_dim = out_dim
        
    def forward(self, graph, node_feat, edge_feat=None):
        """
        Forward pass.
        """
        # Standard GAT
        h, attn = self.gat(graph, node_feat, get_attention=True)
        
        # Flatten multi-head output
        if h.dim() == 3:
            h = h.flatten(1)
        
        # Fuse edge features if available
        if edge_feat is not None and self.edge_encoder is not None:
            with graph.local_scope():
                # Encode edge features
                edge_emb = self.edge_encoder(edge_feat)
                
                # Aggregate edge features to nodes
                graph.ndata['h'] = h
                graph.edata['e'] = edge_emb
                
                # Message passing: aggregate edge features
                graph.update_all(
                    fn.copy_e('e', 'm'),
                    fn.mean('m', 'edge_agg')
                )
                
                # Fuse node and aggregated edge features
                h_combined = torch.cat([h, graph.ndata['edge_agg']], dim=-1)
                h = self.fusion(h_combined)
        
        return h, attn


class AdaptiveEdgeGATLayer(nn.Module):
    """
    Adaptive edge-aware GAT layer.

    Strategy 3: use a gating mechanism to adaptively fuse edge features.
    """
    def __init__(self, in_dim, out_dim, edge_dim, num_heads=4,
                 feat_drop=0.1, attn_drop=0.1):
        super().__init__()
        
        # GAT for node features
        self.node_gat = GATv2Conv(in_dim, out_dim // num_heads,
                                 num_heads=num_heads,
                                 feat_drop=feat_drop,
                                 attn_drop=attn_drop,
                                 allow_zero_in_degree=True)
        
        if edge_dim > 0:
            # Edge feature encoding
            self.edge_encoder = nn.Linear(edge_dim, out_dim)
            
            # Gate: decide how much edge info to use
            self.gate = nn.Sequential(
                nn.Linear(out_dim * 2, out_dim),
                nn.Sigmoid()
            )
            
            # Output projection
            self.output_proj = nn.Linear(out_dim, out_dim)
        else:
            self.edge_encoder = None
            self.gate = None
            self.output_proj = None
            
        self.num_heads = num_heads
        self.dropout = nn.Dropout(feat_drop)
        
    def forward(self, graph, node_feat, edge_feat=None):
        """
        Forward pass.
        """
        # Node features via GAT
        h_node, attn = self.node_gat(graph, node_feat, get_attention=True)
        
        if h_node.dim() == 3:
            h_node = h_node.flatten(1)
        
        # If no edge features, return early
        if edge_feat is None or self.edge_encoder is None:
            return h_node, attn
        
        with graph.local_scope():
            # Encode edge features
            edge_emb = self.edge_encoder(edge_feat)
            edge_emb = F.elu(edge_emb)
            
            # Aggregate edges to nodes
            graph.edata['e'] = edge_emb
            graph.ndata['h'] = h_node
            
            # Max aggregation to keep the most salient edge information
            graph.update_all(
                fn.copy_e('e', 'm'),
                fn.max('m', 'edge_max')
            )
            
            # Gated fusion
            h_concat = torch.cat([h_node, graph.ndata['edge_max']], dim=-1)
            gate = self.gate(h_concat)
            
            # Adaptive fusion via gates
            h_fused = gate * h_node + (1 - gate) * graph.ndata['edge_max']
            
            # Final projection
            h_out = self.output_proj(h_fused)
            h_out = self.dropout(h_out)
            
            return h_out, attn


class MultiScaleEdgeGATLayer(nn.Module):
    """
    Multi-scale edge-aware GAT layer.

    Strategy 4: process edge features at multiple scales.
    """
    def __init__(self, in_dim, out_dim, edge_dim, num_heads=4,
                 feat_drop=0.1, attn_drop=0.1):
        super().__init__()
        
        # Main node GAT
        self.gat = GATv2Conv(in_dim, out_dim // num_heads,
                            num_heads=num_heads,
                            feat_drop=feat_drop,
                            attn_drop=attn_drop,
                            allow_zero_in_degree=True)
        
        if edge_dim > 0:
            # Two branches for edge feature extraction
            self.edge_conv1 = nn.Conv1d(edge_dim, out_dim // 2, kernel_size=1)
            self.edge_conv2 = nn.Conv1d(edge_dim, out_dim // 2, kernel_size=1)
            
            # Fuse multi-scale edge features
            self.edge_fusion = nn.Linear(out_dim, out_dim)
            
            # Final fusion (node + aggregated edge)
            self.final_fusion = nn.Sequential(
                nn.Linear(out_dim * 2, out_dim),
                nn.LayerNorm(out_dim),
                nn.ELU(),
                nn.Dropout(feat_drop)
            )
        else:
            self.edge_conv1 = None
            
        self.num_heads = num_heads
        
    def forward(self, graph, node_feat, edge_feat=None):
        """
        Forward pass.
        """
        # Node path via GAT
        h, attn = self.gat(graph, node_feat, get_attention=True)
        
        if h.dim() == 3:
            h = h.flatten(1)
        
        if edge_feat is None or self.edge_conv1 is None:
            return h, attn
        
        with graph.local_scope():
            # Multi-scale edge feature processing
            E = edge_feat.shape[0]
            edge_feat_reshaped = edge_feat.unsqueeze(0).transpose(1, 2)  # [1, edge_dim, E]
            
            # Two different 1x1 convolutions
            edge_scale1 = self.edge_conv1(edge_feat_reshaped).transpose(1, 2).squeeze(0)  # [E, out_dim//2]
            edge_scale2 = self.edge_conv2(edge_feat_reshaped).transpose(1, 2).squeeze(0)  # [E, out_dim//2]
            
            # Concatenate scales
            edge_multi = torch.cat([edge_scale1, edge_scale2], dim=-1)  # [E, out_dim]
            edge_multi = self.edge_fusion(edge_multi)
            edge_multi = F.elu(edge_multi)
            
            # Aggregate edges to nodes
            graph.edata['e_multi'] = edge_multi
            graph.ndata['h'] = h
            
            graph.update_all(
                fn.copy_e('e_multi', 'm'),
                fn.mean('m', 'edge_agg')
            )
            
            # Fuse node and aggregated edge representations
            h_final = torch.cat([h, graph.ndata['edge_agg']], dim=-1)
            h_final = self.final_fusion(h_final)
            
            return h_final, attn
