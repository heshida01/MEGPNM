import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATv2Conv, GlobalAttentionPooling
from dgl.nn.pytorch import AvgPooling, MaxPooling, SumPooling
import dgl.function as fn


class JumpingKnowledge(nn.Module):
    def __init__(self, mode='concat', hidden_dim=256, num_layers=3):
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        if mode == 'concat':
            self.proj = nn.Linear(hidden_dim * num_layers, hidden_dim)


    def forward(self, layer_outputs):
        if self.mode == 'concat':
            output = torch.cat(layer_outputs, dim=-1)
            return self.proj(output)


class EnhancedGATModel(nn.Module):
    def __init__(self, input_dim=78, edge_dim=9, hidden_dim=256, num_layers=3,
                 heads=4, dropout=0.1, use_edge_feat=True, jk_mode='concat',
                 pooling='gap', edge_fusion_type='standard'):
        super().__init__()

        assert hidden_dim % heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_edge_feat = use_edge_feat
        self.jk_mode = jk_mode
        self.pooling = pooling
        self.edge_fusion_type = edge_fusion_type

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_projection_edge = None

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()


        edge_gat_classes = {
            'multiscale': MultiScaleEdgeGATLayer
        }

        EdgeGATClass = edge_gat_classes[edge_fusion_type]

        for _ in range(num_layers):
            self.convs.append(EdgeGATClass(hidden_dim, hidden_dim, edge_dim,
                                            num_heads=heads, feat_drop=dropout,
                                            attn_drop=dropout))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))


        self.jk = JumpingKnowledge(mode=jk_mode, hidden_dim=hidden_dim, num_layers=num_layers)

        if pooling == 'hybrid':
            self.jk_ms_fusion = JKMultiScaleFusion(
                in_dim_jk=hidden_dim,
                in_dim_scale=hidden_dim,
                d_model=hidden_dim,
                num_heads=heads,
                mode='gate',
                dropout=dropout
            )

        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.gap = GlobalAttentionPooling(gate_nn)

        self.mean_pool = AvgPooling()
        self.max_pool = MaxPooling()
        self.sum_pool = SumPooling()

        self.dropout = nn.Dropout(dropout)

        final_dim = hidden_dim + hidden_dim
        classifier_input_dim = final_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.node_attention_weights = None
        self.edge_attention_weights = []
        self.jk_ms_weights = None

        self.last_gat_output = None
        self.pre_pool_output = None

    def forward(self, graph, return_attention=False):
        node_feat = graph.ndata['feat']
        edge_feat = graph.edata.get('feat', None) if self.use_edge_feat else None

        x = self.input_projection(node_feat)
        x = F.elu(x)
        if edge_feat is not None and self.input_projection_edge is not None:
            edge_feat = F.elu(self.input_projection_edge(edge_feat))

        layer_outputs = []
        multi_scale_features = []
        per_layer_multi = []
        self.edge_attention_weights = []

        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            if self.use_edge_feat and edge_feat is not None:
                x, attn = conv(graph, x, edge_feat)
                self.edge_attention_weights.append(attn)
            else:
                raise ValueError("EdgeGATLayer requires edge features when use_edge_feat=True")

            if x.dim() == 3:
                x = x.flatten(1)

            x = bn(x)
            x = F.elu(x)
            x = self.dropout(x)

            layer_outputs.append(x)

            x_mean = self.mean_pool(graph, x)
            x_max = self.max_pool(graph, x)
            x_sum = self.sum_pool(graph, x)
            per_layer_multi.append(torch.stack([x_mean, x_max, x_sum], dim=1))

        if layer_outputs:
            self.last_gat_output = layer_outputs[-1]
        else:
            self.last_gat_output = x

        if self.jk_mode != 'none':
            x_jk = self.jk(layer_outputs)
        self.pre_pool_output = x_jk


        g_jk = self.gap(graph, x_jk)

        if hasattr(self, 'gap'):
            with torch.no_grad():
                self.node_attention_weights = torch.sigmoid(self.gap.gate_nn(x_jk))

        multi_feats = torch.stack(per_layer_multi, dim=1)

        fused, gates_or_attn = self.jk_ms_fusion(g_jk, multi_feats)

        self.jk_ms_weights = gates_or_attn

        graph_emb = torch.cat([g_jk, fused], dim=-1)


        output = self.classifier(graph_emb)

        if return_attention:
            return output, self.node_attention_weights, self.edge_attention_weights
        return output

    def get_node_attention(self, graph):
        if self.node_attention_weights is None:
            with torch.no_grad():
                _ = self.forward(graph, return_attention=True)
        return self.node_attention_weights

    def get_edge_attention(self, graph):
        if not self.edge_attention_weights:
            with torch.no_grad():
                _ = self.forward(graph, return_attention=True)

        if self.edge_attention_weights and len(self.edge_attention_weights) > 0:
            valid_weights = [w for w in self.edge_attention_weights if w is not None]
            if valid_weights:
                avg_attention = torch.stack(valid_weights).mean(dim=0)
                return avg_attention

        return None

    def get_jk_ms_weights(self, graph):
        if self.pooling != 'hybrid':
            return None

        if self.jk_ms_weights is None:
            with torch.no_grad():
                _ = self.forward(graph)

        return self.jk_ms_weights


class JKMultiScaleFusion(nn.Module):
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
                nn.Linear(d_model, 1)
            )

    def forward(self, g_jk, multi_feats):
        B, L, K, Ds = multi_feats.shape
        S = L * K
        P = multi_feats.reshape(B, S, Ds)

        if self.mode == 'gate':
            gj = g_jk.unsqueeze(1).expand(-1, S, -1)
            gates = torch.sigmoid(self.gate_mlp(torch.cat([gj, P], dim=-1)))
            val = self.value(P)
            out = (gates * val).sum(dim=1)
            return self.proj_out(out), gates.squeeze(-1)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")


class MultiScaleEdgeGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, num_heads=4,
                 feat_drop=0.1, attn_drop=0.1):
        super().__init__()

        self.gat = GATv2Conv(in_dim, out_dim // num_heads,
                            num_heads=num_heads,
                            feat_drop=feat_drop,
                            attn_drop=attn_drop,
                            allow_zero_in_degree=True)

        if edge_dim > 0:
            self.edge_conv1 = nn.Conv1d(edge_dim, out_dim // 2, kernel_size=1)
            self.edge_conv2 = nn.Conv1d(edge_dim, out_dim // 2, kernel_size=1)

            self.edge_fusion = nn.Linear(out_dim, out_dim)

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
        h, attn = self.gat(graph, node_feat, get_attention=True)

        if h.dim() == 3:
            h = h.flatten(1)


        with graph.local_scope():
            E = edge_feat.shape[0]
            if E == 0:
                return h, attn
            edge_feat_reshaped = edge_feat.unsqueeze(0).transpose(1, 2)

            edge_scale1 = self.edge_conv1(edge_feat_reshaped).transpose(1, 2).squeeze(0)
            edge_scale2 = self.edge_conv2(edge_feat_reshaped).transpose(1, 2).squeeze(0)

            edge_multi = torch.cat([edge_scale1, edge_scale2], dim=-1)
            edge_multi = self.edge_fusion(edge_multi)
            edge_multi = F.elu(edge_multi)

            graph.edata['e_multi'] = edge_multi
            graph.ndata['h'] = h

            graph.update_all(
                fn.copy_e('e_multi', 'm'),
                fn.mean('m', 'edge_agg')
            )

            if 'edge_agg' not in graph.ndata:
                edge_agg = torch.zeros_like(h)
            else:
                edge_agg = graph.ndata['edge_agg']

            h_final = torch.cat([h, edge_agg], dim=-1)
            h_final = self.final_fusion(h_final)

            return h_final, attn
