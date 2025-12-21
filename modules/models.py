# -*- coding: utf-8 -*-
"""
Model architectures.

Includes JumpingKnowledge and the EnhancedGATModel.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATv2Conv, GlobalAttentionPooling
from dgl.nn.pytorch import AvgPooling, MaxPooling, SumPooling
import dgl.function as fn



class JumpingKnowledge(nn.Module):
    """
    Jumping Knowledge network to fuse representations from multiple layers.

    Currently supports `concat`.
    """
    def __init__(self, mode='concat', hidden_dim=256, num_layers=3):
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        if mode == 'concat':
            self.proj = nn.Linear(hidden_dim * num_layers, hidden_dim)

            
    def forward(self, layer_outputs):
        """
        Args:
            layer_outputs: List[Tensor], node embeddings from each layer.
        Returns:
            Fused node embeddings.
        """
        if self.mode == 'concat':
            ###layer_outputs shape [4,18402,512]
            output = torch.cat(layer_outputs, dim=-1) #shape (18402, 2048)
            return self.proj(output)
        


class EnhancedGATModel(nn.Module):  # model entry
    """
    Enhanced GAT model.

    - Supports edge features
    - Jumping Knowledge fusion
    - Global Attention Pooling
    - Multi-pooling / multi-scale aggregation
    """
    def __init__(self, input_dim=78, edge_dim=9, hidden_dim=256, num_layers=3, 
                 heads=4, dropout=0.1, use_edge_feat=True, jk_mode='concat',
                 pooling='gap', edge_fusion_type='standard'):
        super().__init__()
        
        # Parameter validation
        assert hidden_dim % heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"
        
        self.hidden_dim = hidden_dim #512
        self.num_layers = num_layers #4
        self.use_edge_feat = use_edge_feat #True
        self.jk_mode = jk_mode #'concat'
        self.pooling = pooling #'hybrid'
        self.edge_fusion_type = edge_fusion_type #'multiscale'
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        # Only create edge projection when needed.
        # Note: for edge-aware layers below, pre-projecting edge features is not required.
        # if use_edge_feat and edge_dim > 0 and edge_fusion_type == 'standard':
        #     self.input_projection_edge = nn.Linear(edge_dim, edge_dim*4)
        # else:
        self.input_projection_edge = None
        
        # GAT layers (edge-aware)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        
        #if edge_fusion_type in [ 'multiscale']:
        # Edge-aware layer registry

        edge_gat_classes = {
            'multiscale': MultiScaleEdgeGATLayer  # multi-scale fusion
        }
        
        EdgeGATClass = edge_gat_classes[edge_fusion_type]
        
        for _ in range(num_layers):
            self.convs.append(EdgeGATClass(hidden_dim, hidden_dim, edge_dim,
                                            num_heads=heads, feat_drop=dropout,
                                            attn_drop=dropout))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        
        # Jumping Knowledge
        self.jk = JumpingKnowledge(mode=jk_mode, hidden_dim=hidden_dim, num_layers=num_layers)
        
        # JK-MultiScale fusion (for hybrid pooling)
        if pooling == 'hybrid':
            self.jk_ms_fusion = JKMultiScaleFusion(
                in_dim_jk=hidden_dim, 
                in_dim_scale=hidden_dim,
                d_model=hidden_dim, 
                num_heads=heads, 
                mode='gate',  # 'gate' or 'coattn'
                dropout=dropout
            )
        
        # Pooling layers
        #if pooling in ['gap', 'hybrid']:  # hybrid also needs GAP
            # Global Attention Pooling
        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.gap = GlobalAttentionPooling(gate_nn)
        
        # Traditional graph-level pooling
        self.mean_pool = AvgPooling()
        self.max_pool = MaxPooling()
        self.sum_pool = SumPooling()
        
        self.dropout = nn.Dropout(dropout)
        
        # Output dimension
        # if pooling == 'gap':
        #     final_dim = hidden_dim
        # elif pooling == 'hybrid':
        #     # JK graph summary + fused interaction
        #     final_dim = hidden_dim + hidden_dim
        # elif pooling == 'multi':
        #     # Multi-scale: 3 pooling ops per layer
        #     final_dim = num_layers * 3 * hidden_dim
        # else:
        #     final_dim = 3 * hidden_dim  # mean + max + sum
        final_dim = hidden_dim + hidden_dim
        classifier_input_dim = final_dim
        
        # Regressor head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Saved attention weights for explainability
        self.node_attention_weights = None
        self.edge_attention_weights = []
        self.jk_ms_weights = None  # JK-MultiScale interaction weights
        
        # Intermediate activations for Grad-CAM
        self.last_gat_output = None
        self.pre_pool_output = None
        
    def forward(self, graph, return_attention=False):
        """
        Forward pass.

        Args:
            graph: DGL graph.
            return_attention: Whether to return attention weights.
        """
        # Fetch features
        node_feat = graph.ndata['feat'] # shape: (18402, 110)
        edge_feat = graph.edata.get('feat', None) if self.use_edge_feat else None #shape 39658,9
        
        # Input projection
        x = self.input_projection(node_feat)
        x = F.elu(x)
        if edge_feat is not None and self.input_projection_edge is not None:
            edge_feat = F.elu(self.input_projection_edge(edge_feat))
        
        # Collect per-layer outputs
        layer_outputs = []
        multi_scale_features = []
        per_layer_multi = []  # for hybrid: list of [B, 3, D]
        self.edge_attention_weights = []
        
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            if self.use_edge_feat and edge_feat is not None:
                x, attn = conv(graph, x, edge_feat)
                self.edge_attention_weights.append(attn)
            else:
                raise ValueError("EdgeGATLayer requires edge features when use_edge_feat=True")
                
            if x.dim() == 3:
                # Flatten multi-head output: (N, heads, out_dim) -> (N, heads * out_dim)
                x = x.flatten(1)
            
            x = bn(x)
            x = F.elu(x)
            x = self.dropout(x)
            
            layer_outputs.append(x)
            
            # Multi-pooling per layer (mean / max / sum)
            x_mean = self.mean_pool(graph, x)
            x_max = self.max_pool(graph, x)
            x_sum = self.sum_pool(graph, x)
            per_layer_multi.append(torch.stack([x_mean, x_max, x_sum], dim=1))  
        
        # Save last GAT output (for Grad-CAM)
        if layer_outputs:
            self.last_gat_output = layer_outputs[-1]
        else:
            self.last_gat_output = x
        
        # Jumping Knowledge fusion
        if self.jk_mode != 'none':
            x_jk = self.jk(layer_outputs)
        # else:
        #     x_jk = layer_outputs[-1] if layer_outputs else x   
        # Save pre-pooling output (for Grad-CAM)
        self.pre_pool_output = x_jk
        
        # Graph-level pooling

        #elif self.pooling == 'hybrid':
        # JK summary via GAP
        g_jk = self.gap(graph, x_jk)# if hasattr(self, 'gap') else self.mean_pool(graph, x_jk)
        
        # Manually compute node weights as attention scores
        if hasattr(self, 'gap'):
            with torch.no_grad():
                self.node_attention_weights = torch.sigmoid(self.gap.gate_nn(x_jk)) #x_jk shape (18402, 512) self.node_attention_weights shape (18402, 1)
        
        # Stack multi-pooling features across layers: [B, L, 3, D]
        multi_feats = torch.stack(per_layer_multi, dim=1) #multi_feats shape (512 4, 3, 512)
        
        # JK-MultiScale fusion
        fused, gates_or_attn = self.jk_ms_fusion(g_jk, multi_feats)# shape fused (512,512), gates_or_attn(512,12)
        
        # Save interaction weights for explainability
        self.jk_ms_weights = gates_or_attn  # [B, L*3]
        
        # Concatenate JK summary and fused multi-scale features
        graph_emb = torch.cat([g_jk, fused], dim=-1)# shape (512,1024)

        
        # Prediction
        output = self.classifier(graph_emb)
        
        if return_attention:
            return output, self.node_attention_weights, self.edge_attention_weights
        return output
    
    def get_node_attention(self, graph):
        """Get node attention weights."""
        if self.node_attention_weights is None:
            # Run a forward pass once to populate weights
            with torch.no_grad():
                _ = self.forward(graph, return_attention=True)
        return self.node_attention_weights
    
    def get_edge_attention(self, graph):
        """Get edge attention weights averaged over layers."""
        if not self.edge_attention_weights:
            with torch.no_grad():
                _ = self.forward(graph, return_attention=True)
        
        # Average attention weights across layers
        if self.edge_attention_weights and len(self.edge_attention_weights) > 0:
            # Ensure tensors exist and are non-empty
            valid_weights = [w for w in self.edge_attention_weights if w is not None]
            if valid_weights:
                avg_attention = torch.stack(valid_weights).mean(dim=0)
                return avg_attention
        
        # If no valid edge attention exists, return None.
        return None
    
    def get_jk_ms_weights(self, graph):
        """Get JK-MultiScale interaction weights (hybrid pooling only)."""
        if self.pooling != 'hybrid':
            return None
        
        if self.jk_ms_weights is None:
            # Run a forward pass once to populate weights
            with torch.no_grad():
                _ = self.forward(graph)
        
        return self.jk_ms_weights  # [B, L*3]


class JKMultiScaleFusion(nn.Module):
    """
    Fuse multi-layer multi-pooling graph features with a JK summary.

    mode='gate': element-wise gating over (layer × pooling) features.
    mode='coattn': cross-layer/cross-scale attention (not enabled here).
    """
    def __init__(self, in_dim_jk, in_dim_scale, d_model=256, num_heads=4, mode='gate', dropout=0.1):
        super().__init__()
        self.mode = mode
        self.proj_out = nn.Linear(d_model, d_model)

        if mode == 'gate':
            self.value = nn.Linear(in_dim_scale, d_model)
            self.gate_mlp = nn.Sequential(
                nn.Linear(in_dim_jk + in_dim_scale, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1)  # scalar gate
            )

    def forward(self, g_jk, multi_feats):
        """
        Args:
            g_jk: [B, Dj], JK graph-level summary (e.g., GAP(x_jk)).
            multi_feats: [B, L, 3, Ds], per-layer graph features from 3 pooling ops.

        Returns:
            fused: [B, d_model] fused representation.
            weights: explainability weights (e.g., gates) with shape [B, L*3].
        """
        B, L, K, Ds = multi_feats.shape # 512 4 3 512
        S = L * K
        P = multi_feats.reshape(B, S, Ds)  # [B, S, Ds] # 512 12 512

        if self.mode == 'gate':
            gj = g_jk.unsqueeze(1).expand(-1, S, -1)             # [B, S, Dj]
            gates = torch.sigmoid(self.gate_mlp(torch.cat([gj, P], dim=-1)))  # [B, S, 1]
            val = self.value(P)                                   # [B, S, d_model]
            out = (gates * val).sum(dim=1)                        # [B, d_model]
            return self.proj_out(out), gates.squeeze(-1)          # return gates for explainability
        # elif self.mode == 'coattn':
        #     q = self.q(g_jk).unsqueeze(1)                         # [B, 1, d_model]
        #     k = self.k(P)                                         # [B, S, d_model]
        #     v = self.v(P)                                         # [B, S, d_model]
        #     out, attn = self.attn(q, k, v)                        # out:[B,1,d], attn:[B,1,S]
        #     return self.proj_out(out.squeeze(1)), attn.squeeze(1) # [B,S]
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")




class MultiScaleEdgeGATLayer(nn.Module):
    """
    Multi-scale edge-aware GAT layer.

    Processes edge features at multiple scales and aggregates them back to nodes.
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
        
        # If edge branch is disabled, return the node-path output.
        # if edge_feat is None or self.edge_conv1 is None:
        #     return h, attn
        # # If there are no edges, treat edge aggregation as zeros and run final_fusion.
        # if graph.num_edges() == 0:
        #     with torch.no_grad():
        #         edge_agg = torch.zeros_like(h)
        #     h_final = torch.cat([h, edge_agg], dim=-1)
        #     h_final = self.final_fusion(h_final)
        #     return h_final, attn
        
        with graph.local_scope():
            # Multi-scale edge feature processing
            E = edge_feat.shape[0]
            if E == 0:
                return h, attn
            edge_feat_reshaped = edge_feat.unsqueeze(0).transpose(1, 2)  # [1, edge_dim, E]
            
            # Two different 1x1 convolutions
            edge_scale1 = self.edge_conv1(edge_feat_reshaped).transpose(1, 2).squeeze(0)  # [E, out_dim//2]
            edge_scale2 = self.edge_conv2(edge_feat_reshaped).transpose(1, 2).squeeze(0)  # [E, out_dim//2]
            
            # Concatenate scales
            edge_multi = torch.cat([edge_scale1, edge_scale2], dim=-1)  # [E, out_dim]
            edge_multi = self.edge_fusion(edge_multi)
            edge_multi = F.elu(edge_multi)
            
            # Aggregate edges to nodes
            graph.edata['e_multi'] = edge_multi #shape (39658, 512)
            graph.ndata['h'] = h #shape (18402, 512)
            
            graph.update_all(
                fn.copy_e('e_multi', 'm'),
                fn.mean('m', 'edge_agg')
            )
            
            # If edge aggregation is missing, fall back to zeros.
            if 'edge_agg' not in graph.ndata:
                edge_agg = torch.zeros_like(h)
            else:
                edge_agg = graph.ndata['edge_agg']
            
            # Fuse node and aggregated edge representations
            h_final = torch.cat([h, edge_agg], dim=-1)
            h_final = self.final_fusion(h_final)
            
            return h_final, attn
