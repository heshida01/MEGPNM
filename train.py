import os
import sys
import json
import argparse
import random
from datetime import datetime

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

from modules import (
    set_random_seed,
    ENHANCED_ATOM_FEATURE_DIM,
    validate_atom_features,
    GraphDataset,
    graph_collate_fn,
    EnhancedGATModel,
    train_model,
)


def _boolean_action_supported():
    try:
        _ = argparse.BooleanOptionalAction
        return True
    except Exception:
        return False


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ('y', 'yes', 't', 'true', '1'):
        return True
    if s in ('n', 'no', 'f', 'false', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean expected.')


def enforce_strict_determinism():
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(device_arg: str):
    if device_arg != 'auto':
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def seed_worker(worker_id, base_seed: int):
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def validate_required_columns(df: pd.DataFrame, required_cols, name: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required column(s): {missing}")


def maybe_limit_df(df: pd.DataFrame, limit: int):
    if limit is not None and limit > 0:
        return df.iloc[:limit].reset_index(drop=True)
    return df


def _pkg_version(module_name):
    try:
        import importlib
        return getattr(importlib.import_module(module_name), '__version__', None)
    except Exception:
        return None


def save_env_info(device):
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn": (torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None),
        "dgl": _pkg_version('dgl'),
        "rdkit": _pkg_version('rdkit'),
        "numpy": _pkg_version('numpy'),
        "pandas": _pkg_version('pandas'),
        "scikit_learn": _pkg_version('sklearn'),
        "scipy": _pkg_version('scipy'),
        "device": str(device),
        "time": datetime.now().isoformat(timespec="seconds"),
    }

    if device.type == 'cuda' and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        try:
            info["gpu_name"] = torch.cuda.get_device_name(idx)
            major, minor = torch.cuda.get_device_capability(idx)
            info["gpu_capability"] = f"{major}.{minor}"
        except Exception:
            pass
        info["cuda_visible_devices"] = os.environ.get('CUDA_VISIBLE_DEVICES')
        info["torch_device_index"] = idx

    return info


def collect_split_predictions(model, data_loader, device, true_decimals=3):
    model.eval()
    smiles, preds, targets = [], [], []

    with torch.no_grad():
        for batch in data_loader:
            batch_graph = batch['graph'].to(device)
            outputs = model(batch_graph)

            batch_preds = outputs.detach().cpu().view(-1).tolist()
            batch_targets = batch['labels'].detach().cpu().view(-1).tolist()

            preds.extend(batch_preds)
            targets.extend(batch_targets)
            smiles.extend(batch['smiles'])

    df = pd.DataFrame({
        'smiles': smiles,
        'true_value': targets,
        'predicted_value': preds
    })
    if true_decimals is not None:
        df['true_value'] = df['true_value'].round(true_decimals)
    df['error'] = df['predicted_value'] - df['true_value']
    df['abs_error'] = df['error'].abs()
    return df


def build_parser():
    parser = argparse.ArgumentParser(
        description='MEGPNM: multi-scale edge-aware GAT with hybrid pooling')

    parser.add_argument('--data_dir', type=str, default='dataset/random',
                        help='Directory containing train.csv, val.csv, test.csv')
    parser.add_argument('--epochs', type=int, default=180)
    parser.add_argument('--lr', type=float, default=8.5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='AdamW weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient-norm clipping; 0 disables it. '
                             'Raising this materially degrades results.')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--eval_batch_size', type=int, default=None,
                        help='If set, overrides batch size for val/test')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--random_seed', type=int, default=42)

    parser.add_argument('--save_dir', type=str, default='runs/default',
                        help='Directory to store checkpoints, logs, and results')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint to load for prediction export; '
                             'defaults to <save_dir>/best_model.pth')
    parser.add_argument('--skip_train', action='store_true',
                        help='Skip training and only export predictions from the checkpoint')
    parser.add_argument('--no_save_model', action='store_true',
                        help='Disable saving the best model during training')

    if _boolean_action_supported():
        parser.add_argument('--use_edge_feat', default=True, action=argparse.BooleanOptionalAction,
                            help='Whether to use edge features')
    else:
        parser.add_argument('--use_edge_feat', type=str2bool, default=True,
                            help='Whether to use edge features')

    parser.add_argument('--jk_mode', type=str, default='concat',
                        choices=['concat'],
                        help='Jumping Knowledge mode')
    parser.add_argument('--pooling', type=str, default='hybrid',
                        choices=['hybrid'],
                        help='Graph readout')
    parser.add_argument('--edge_fusion_type', type=str, default='multiscale',
                        choices=['multiscale'],
                        help='Edge feature fusion type')

    parser.add_argument('--deterministic', action='store_true',
                        help='Force deterministic algorithms (may reduce speed)')
    parser.add_argument('--allow_tf32', action='store_true',
                        help='Allow TF32 on Ampere+ GPUs for speed (slight precision change)')

    parser.add_argument('--limit_train_samples', type=int, default=0,
                        help='If >0, limit train rows for quick tests')
    parser.add_argument('--limit_eval_samples', type=int, default=0,
                        help='If >0, limit val/test rows for quick tests')

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    print(args)

    set_random_seed(args.random_seed)
    if args.deterministic:
        enforce_strict_determinism()

    device = get_device(args.device)
    print("\n" + "=" * 80)
    print("Enhanced atom features validation")
    print("=" * 80)
    validate_atom_features()
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32

    os.makedirs(args.save_dir, exist_ok=True)

    print("=== Loading data ===")
    train_csv = os.path.join(args.data_dir, 'train.csv')
    val_csv = os.path.join(args.data_dir, 'val.csv')
    test_csv = os.path.join(args.data_dir, 'test.csv')
    if not (os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv)):
        raise FileNotFoundError(f"Expected train/val/test CSVs under {args.data_dir}")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    required_cols = ['smiles', 'standardized_value']
    validate_required_columns(train_df, required_cols, 'train.csv')
    validate_required_columns(val_df, required_cols, 'val.csv')
    validate_required_columns(test_df, required_cols, 'test.csv')

    if args.limit_train_samples and args.limit_train_samples > 0:
        train_df = maybe_limit_df(train_df, args.limit_train_samples)
    if args.limit_eval_samples and args.limit_eval_samples > 0:
        val_df = maybe_limit_df(val_df, args.limit_eval_samples)
        test_df = maybe_limit_df(test_df, args.limit_eval_samples)

    train_dataset = GraphDataset(
        train_df['smiles'].tolist(),
        train_df['standardized_value'].values,
        use_edge_feat=args.use_edge_feat
    )
    val_dataset = GraphDataset(
        val_df['smiles'].tolist(),
        val_df['standardized_value'].values,
        use_edge_feat=args.use_edge_feat
    )
    test_dataset = GraphDataset(
        test_df['smiles'].tolist(),
        test_df['standardized_value'].values,
        use_edge_feat=args.use_edge_feat
    )

    g = torch.Generator()
    g.manual_seed(args.random_seed)

    is_cuda = (device.type == 'cuda')
    if args.deterministic:
        num_workers = 0
        persistent = False
    else:
        num_workers = 0 if not is_cuda else max(1, min(8, (os.cpu_count() or 2) - 1))
        persistent = is_cuda and num_workers > 0

    pin_memory = is_cuda
    eval_bs = args.eval_batch_size or args.batch_size

    worker_fn = (lambda wid: seed_worker(wid, args.random_seed)) if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=graph_collate_fn,
        num_workers=num_workers,
        generator=g,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent,
        worker_init_fn=worker_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_bs,
        shuffle=False,
        collate_fn=graph_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent,
        worker_init_fn=worker_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_bs,
        shuffle=False,
        collate_fn=graph_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent,
        worker_init_fn=worker_fn
    )

    print(f"Train set: {len(train_dataset)} | Val set: {len(val_dataset)} | Test set: {len(test_dataset)}")

    set_random_seed(args.random_seed)

    edge_dim = 9 if args.use_edge_feat else 0
    model = EnhancedGATModel(
        input_dim=ENHANCED_ATOM_FEATURE_DIM,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_edge_feat=args.use_edge_feat,
        jk_mode=args.jk_mode,
        pooling=args.pooling,
        edge_fusion_type=args.edge_fusion_type
    ).to(device)
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params:,}")

    init_model_path = os.path.join(args.save_dir, 'init_model.pth')
    try:
        torch.save(model.state_dict(), init_model_path)
        print(f"[INFO] Saved randomly initialized model to: {init_model_path}")
    except Exception as e:
        print(f"[WARN] Failed to save initial model: {e}")

    best_results = None
    if not args.skip_train:
        best_results, _ = train_model(model, train_loader, val_loader, test_loader, args, device)
    else:
        print("[INFO] --skip_train set. Skipping training.")

    final_results = {
        'args': vars(args),
        'best_results': best_results,
        'random_seed': args.random_seed,
        'device': str(device),
        'deterministic_mode': bool(args.deterministic),
        'env': save_env_info(device),
    }

    candidate_ckpts = []
    if args.checkpoint:
        candidate_ckpts.append(args.checkpoint)
    candidate_ckpts.append(os.path.join(args.save_dir, 'best_model.pth'))

    ckpt_path = None
    for p in candidate_ckpts:
        if p and os.path.exists(p):
            ckpt_path = p
            break

    if ckpt_path:
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()
            print(f"[INFO] Loaded checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint at {ckpt_path}: {e}")
            ckpt_path = None
    else:
        print("[WARN] No checkpoint found; skipping prediction export.")

    prediction_artifacts = {}
    if ckpt_path:
        for split_name, loader in (('val', val_loader), ('test', test_loader)):
            try:
                pred_df = collect_split_predictions(model, loader, device)
                pred_path = os.path.join(args.save_dir, f'best_model_{split_name}_predictions.csv')
                pred_df.to_csv(pred_path, index=False)
                prediction_artifacts[f'{split_name}_predictions'] = pred_path
                print(f"[INFO] Saved {split_name} predictions to {pred_path}")
            except Exception as e:
                print(f"[WARN] Failed to save {split_name} predictions: {e}")

    if prediction_artifacts:
        final_results['prediction_artifacts'] = prediction_artifacts

    print("\n" + "=" * 80)
    print("Final results summary")
    print("=" * 80)

    if final_results['best_results'] is not None:
        br = final_results['best_results']
        val = br.get('val', {})
        test = br.get('test', None)
        try:
            print(f"Val  - RMSE: {val.get('rmse', float('nan')):.4f} | "
                  f"R²: {val.get('r2', float('nan')):.4f} | "
                  f"CI: {val.get('ci', float('nan')):.4f} | "
                  f"PCC: {val.get('pcc', float('nan')):.4f} | "
                  f"SCC: {val.get('scc', float('nan')):.4f}")
            if isinstance(test, dict) and test:
                print(f"Test - RMSE: {test.get('rmse', float('nan')):.4f} | "
                      f"R²: {test.get('r2', float('nan')):.4f} | "
                      f"CI: {test.get('ci', float('nan')):.4f} | "
                      f"PCC: {test.get('pcc', float('nan')):.4f} | "
                      f"SCC: {test.get('scc', float('nan')):.4f}")
        except Exception:
            print("[WARN] Could not pretty-print metrics; raw best_results will be saved.")
    else:
        print("[INFO] No best_results (training skipped or failed).")

    final_path = os.path.join(args.save_dir, 'final_results.json')
    try:
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        print(f"\nTraining complete. Results saved to {final_path}")
    except Exception as e:
        print(f"[WARN] Failed to write final_results.json: {e}")

    print("\nReproducibility notes:")
    print("  1) Use the same --random_seed")
    print("  2) Keep model hyperparameters and data identical")
    print("  3) Run on the same hardware/driver stack")
    print("  4) Match CUDA and PyTorch versions")
    print(f"  5) Current random seed: {args.random_seed}")


if __name__ == '__main__':
    main()
