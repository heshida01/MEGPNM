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

    set_random_seed(args.random_seed)

    save_dir = getattr(args, 'save_dir', '.')
    os.makedirs(save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(save_dir, 'best_model.pth')

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=getattr(args, 'weight_decay', 1e-4))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.8)
    best_val_rmse = float('inf')

    train_log = []

    best_results = None

    print("\n=== Starting MEGPNM training ===")
    print("Edge features: {}".format('Enabled' if args.use_edge_feat else 'Disabled'))
    print("JK mode: {}".format(args.jk_mode))
    print("Pooling: {}".format(args.pooling))
    print("Checkpoint dir: {}".format(save_dir))

    for epoch in range(args.epochs):

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

            grad_clip = getattr(args, 'grad_clip', 1.0)
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        val_results = evaluate_model(model, val_loader, device, prefix="Val")

        train_loss /= num_batches

        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_rmse': val_results['rmse'],
            'val_r2': val_results['r2'],
            'val_ci': val_results['ci'],
            'val_pcc': val_results['pcc'],
            'val_scc': val_results['scc']
        }
        train_log.append(log_entry)

        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val RMSE: {val_results['rmse']:.4f} R²: {val_results['r2']:.4f} "
              f"CI: {val_results['ci']:.4f} PCC: {val_results['pcc']:.4f} SCC: {val_results['scc']:.4f}")

        if val_results['rmse'] < best_val_rmse:
            best_val_rmse = val_results['rmse']
            if not getattr(args, 'no_save_model', False):
                torch.save(model.state_dict(), best_ckpt_path)
            best_results = {
                'val': val_results
            }

        scheduler.step(val_results['rmse'])

    pd.DataFrame(train_log).to_csv(os.path.join(save_dir, 'train_log.csv'), index=False)

    if best_results is not None and not getattr(args, 'no_save_model', False) and os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        test_results = evaluate_model(model, test_loader, device, prefix="Test")
        best_results['test'] = test_results
        print(f"\nTest set results (best checkpoint):")
        print(f"  RMSE: {test_results['rmse']:.4f} | R²: {test_results['r2']:.4f} | "
              f"CI: {test_results['ci']:.4f} | PCC: {test_results['pcc']:.4f} | SCC: {test_results['scc']:.4f}")

    return best_results, train_log


def evaluate_model(model, data_loader, device, prefix=""):
    model.eval()
    predictions = []
    targets = []

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    with torch.no_grad():
        for batch in data_loader:
            batch_graph = batch['graph'].to(device)
            batch_labels = batch['labels'].to(device)

            output = model(batch_graph)

            output_np = output.squeeze().cpu().numpy()
            targets_np = batch_labels.squeeze().cpu().numpy()

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
        'pcc': float(pearson_correlation(targets, predictions)),
        'scc': float(spearman_correlation(targets, predictions))
    }

    return results
