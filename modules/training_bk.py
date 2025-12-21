# -*- coding: utf-8 -*-
"""
Training and evaluation utilities (legacy backup).

This file is kept for reference.
"""
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score

from .utils import set_random_seed, ENHANCED_ATOM_FEATURE_DIM
from .metrics import concordance_index, adjusted_r2, pearson_correlation, spearman_correlation


def train_model(model, train_loader, val_loader, test_loader, args, device='cuda'):
    """Training loop (enhanced version; legacy backup)."""
    
    # Re-seed before training to keep the run deterministic.
    set_random_seed(args.random_seed)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # max LR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0001)  # cosine annealing
    best_val_rmse = float('inf')
    
    # Training log
    train_log = []
    
    print("\n=== Starting Enhanced GAT Model Training ===")
    print("Edge features: {}{}".format('Enabled' if args.use_edge_feat else 'Disabled', ""))
    print("JK mode: {}{}".format(args.jk_mode, ""))
    print("Pooling: {}{}".format(args.pooling, ""))
    print("Deterministic mode: Enabled")
    
    for epoch in range(args.epochs):
        
 
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch in progress_bar:
            batch_graph = batch['graph'].to(device)
            batch_labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            output = model(batch_graph)
            
            loss = criterion(output.squeeze(), batch_labels.squeeze())
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Evaluation
        val_results = evaluate_model(model, val_loader, device, prefix="Val")
        test_results = evaluate_model(model, test_loader, device, prefix="Test")
        
        train_loss /= num_batches
        
        # Log
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_rmse': val_results['rmse'],
            'val_r2': val_results['r2'],
            'val_ci': val_results['ci'],
            # 'val_adj_r2': val_results['adj_r2'],  # adjusted_r2 is not suitable here
            'val_pcc': val_results['pcc'],
            'val_scc': val_results['scc'],
            'test_rmse': test_results['rmse'],
            'test_r2': test_results['r2'],
            'test_ci': test_results['ci'],
            # 'test_adj_r2': test_results['adj_r2'],  # adjusted_r2 is not suitable here
            'test_pcc': test_results['pcc'],
            'test_scc': test_results['scc']
        }
        train_log.append(log_entry)
        
        # Print metrics
        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val RMSE: {val_results['rmse']:.4f} R²: {val_results['r2']:.4f} "
              f"CI: {val_results['ci']:.4f} PCC: {val_results['pcc']:.4f} SCC: {val_results['scc']:.4f} | "
              f"Test RMSE: {test_results['rmse']:.4f} R²: {test_results['r2']:.4f} "
              f"CI: {test_results['ci']:.4f} PCC: {test_results['pcc']:.4f} SCC: {test_results['scc']:.4f}")
        
        # Save best model (unless disabled)
        if val_results['rmse'] < best_val_rmse:
            best_val_rmse = val_results['rmse']
            if not getattr(args, 'no_save_model', False):
                torch.save(model.state_dict(), 'best_model.pth')
            # Save best metrics
            best_results = {
                'val': val_results,
                'test': test_results
            }

        scheduler.step()
        # No early stopping (by design)
    
    # Save training log
    pd.DataFrame(train_log).to_csv('train_log.csv', index=False)
    
    return best_results, train_log


def evaluate_model(model, data_loader, device, prefix=""):
    """Evaluate model performance."""
    model.eval()
    predictions = []
    targets = []
    
    # Keep evaluation deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    with torch.no_grad():
        for batch in data_loader:
            batch_graph = batch['graph'].to(device)
            batch_labels = batch['labels'].to(device)
            
            output = model(batch_graph)
            
            output_np = output.squeeze().cpu().numpy()
            targets_np = batch_labels.squeeze().cpu().numpy()
            
            # Ensure array-like outputs
            if output_np.ndim == 0:
                output_np = [output_np.item()]
            if targets_np.ndim == 0:
                targets_np = [targets_np.item()]
                
            predictions.extend(output_np)
            targets.extend(targets_np)
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    results = {
        'rmse': float(np.sqrt(mean_squared_error(targets, predictions))),
        'r2': float(r2_score(targets, predictions)),
        'ci': float(concordance_index(targets, predictions)),
        # Note: adjusted_r2 is not suitable for this deep learning setup.
        # 'adj_r2': float(adjusted_r2(targets, predictions, ENHANCED_ATOM_FEATURE_DIM)),
        'pcc': float(pearson_correlation(targets, predictions)),
        'scc': float(spearman_correlation(targets, predictions))
    }
    
    return results
