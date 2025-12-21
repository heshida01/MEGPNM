# -*- coding: utf-8 -*-
"""
Dataset and dataloader utilities.

Includes the GraphDataset class and the DGL collate function.
"""
import torch
from torch.utils.data import Dataset
import dgl
from tqdm import tqdm
from .features import smiles_to_dgl_graph


class GraphDataset(Dataset):
    """Molecular graph dataset."""
    def __init__(self, smiles_list, labels, use_edge_feat=True):
        self.smiles_list = smiles_list
        self.labels = torch.tensor(labels, dtype=torch.float)
        self.use_edge_feat = use_edge_feat
        
        print("Converting SMILES to DGL graphs (edge_feat={})...".format(use_edge_feat))
        self.graphs = []
        for i, smiles in enumerate(tqdm(smiles_list)):
            graph = smiles_to_dgl_graph(smiles, use_edge_feat)
            self.graphs.append(graph)
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        graph = self.graphs[idx]
        label = self.labels[idx]
        
        item = {
            'graph': graph,
            'label': label,
            'smiles': self.smiles_list[idx]
        }
            
        return item


def graph_collate_fn(samples):
    """DGL collate function."""
    graphs = [s['graph'] for s in samples]
    labels = torch.stack([s['label'] for s in samples])
    smiles = [s['smiles'] for s in samples]
    
    batched_graph = dgl.batch(graphs)
    
    batch_data = {
        'graph': batched_graph,
        'labels': labels,
        'smiles': smiles
    }
    
    return batch_data
