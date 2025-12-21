# -*- coding: utf-8 -*-

"""Run inference with a trained MEGPNM checkpoint.

Typical usage is to generate predictions for `test.csv` after training.

Expected CSV columns:
- `smiles` (required)
- `standardized_value` (optional; if present, metrics will be reported)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from modules import (
    ENHANCED_ATOM_FEATURE_DIM,
    GraphDataset,
    EnhancedGATModel,
    graph_collate_fn,
    set_random_seed,
)
from modules.metrics import calculate_metrics


def _boolean_action_supported() -> bool:
    try:
        _ = argparse.BooleanOptionalAction
        return True
    except Exception:
        return False


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("y", "yes", "t", "true", "1"):
        return True
    if s in ("n", "no", "f", "false", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean expected.")


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_required_columns(df: pd.DataFrame, required_cols, name: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required column(s): {missing}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MEGPNM checkpoint inference")

    parser.add_argument("--data_csv", type=str, required=True, help="Path to a CSV file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a .pth checkpoint")
    parser.add_argument(
        "--output_csv",
        type=str,
        default="predictions.csv",
        help="Output CSV path (will be overwritten)",
    )

    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument(
        "--target_col",
        type=str,
        default="standardized_value",
        help="Optional target column; if missing, metrics are skipped",
    )

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--random_seed", type=int, default=42)

    # Model hyperparameters (must match training)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument(
        "--jk_mode",
        type=str,
        default="concat",
        choices=["concat"],
        help="Jumping Knowledge mode",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="hybrid",
        choices=["hybrid"],
        help="Pooling method",
    )
    parser.add_argument(
        "--edge_fusion_type",
        type=str,
        default="multiscale",
        choices=["multiscale"],
        help="Edge fusion type",
    )

    if _boolean_action_supported():
        parser.add_argument(
            "--use_edge_feat",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Whether to use edge features",
        )
    else:
        parser.add_argument(
            "--use_edge_feat",
            type=str2bool,
            default=True,
            help="Whether to use edge features",
        )

    return parser


def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch_graph = batch["graph"].to(device)
            outputs = model(batch_graph).detach().cpu().view(-1).numpy()
            preds.extend(outputs.tolist())
    return np.asarray(preds, dtype=np.float32)


def main() -> int:
    args = build_parser().parse_args()

    if not os.path.exists(args.data_csv):
        raise FileNotFoundError(f"CSV not found: {args.data_csv}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    set_random_seed(args.random_seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    df = pd.read_csv(args.data_csv)
    validate_required_columns(df, [args.smiles_col], os.path.basename(args.data_csv))

    has_targets = args.target_col in df.columns
    smiles_list = df[args.smiles_col].astype(str).tolist()
    labels = df[args.target_col].values.astype(np.float32) if has_targets else np.zeros(len(df), dtype=np.float32)

    dataset = GraphDataset(smiles_list, labels, use_edge_feat=args.use_edge_feat)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=graph_collate_fn,
        num_workers=0,
        drop_last=False,
    )

    edge_dim = 9 if args.use_edge_feat else 0
    model = EnhancedGATModel(
        input_dim=ENHANCED_ATOM_FEATURE_DIM,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_edge_feat=args.use_edge_feat,
        jk_mode=args.jk_mode,
        pooling=args.pooling,
        edge_fusion_type=args.edge_fusion_type,
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.checkpoint}")

    preds = predict(model, loader, device)

    out_df = pd.DataFrame(
        {
            "smiles": smiles_list,
            "predicted_value": preds,
        }
    )

    if has_targets:
        true_vals = df[args.target_col].values.astype(np.float32)
        out_df["true_value"] = true_vals
        out_df["error"] = out_df["predicted_value"] - out_df["true_value"]
        out_df["abs_error"] = out_df["error"].abs()

        metrics = calculate_metrics(true_vals, preds)
        print(
            "Metrics - "
            f"RMSE: {metrics.get('rmse', float('nan')):.4f} | "
            f"R²: {metrics.get('r2', float('nan')):.4f} | "
            f"CI: {metrics.get('ci', float('nan')):.4f} | "
            f"PCC: {metrics.get('pcc', float('nan')):.4f} | "
            f"SCC: {metrics.get('scc', float('nan')):.4f}"
        )
    else:
        print(f"Target column '{args.target_col}' not found; skipping metrics.")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

