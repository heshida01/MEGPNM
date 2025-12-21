# MEGPNM: A Multi-Scale Edge-Aware Graph Neural Network for Predicting Membrane Permeability of Nonpeptidic Macrocycles

This repository contains MEGPNM, an enhanced graph neural network (multi-scale edge-aware GAT with Jumping Knowledge + Global Attention Pooling) for **molecular property regression**, with a focus on **PAMPA permeability**.



### Expected data format

`main.py` expects a directory containing the following CSV files:

- `train.csv`
- `val.csv`
- `test.csv`

Each CSV must contain at least:

- `smiles`: SMILES string
- `standardized_value`: regression target (float)

Minimal example:

```csv
smiles,standardized_value
CCO,0.123
```

## Installation

Create a Python environment and install the required packages. At minimum you will need:

- Python 3.9+
- PyTorch
- DGL
- RDKit
- numpy, pandas
- scikit-learn, scipy
- matplotlib, tqdm

(Exact versions depend on your CUDA/OS setup.)

## Training

Run training from the repository root:

```bash
python main.py \
  --data_dir /path/to/split_dir \
  --save_dir runs/exp1 \
  --epochs 180 \
  --batch_size 512 \
  --hidden_dim 512 \
  --num_layers 4
```

During training, each epoch reports **training loss and validation metrics only**.

### Outputs

Artifacts are written under `--save_dir` (e.g. `runs/exp1/`) and/or the working directory:

- `runs/exp1/init_model.pth`: randomly initialized checkpoint
- `best_model.pth`: best checkpoint during training (saved in the working directory by `modules/training.py`)
- `runs/exp1/final_results.json`: run configuration + summary metrics
- `runs/exp1/best_model_val_predictions.csv`, `runs/exp1/best_model_test_predictions.csv`: predictions (if a checkpoint is available)

Tip: if you want to force a specific checkpoint for analysis, pass `--checkpoint /path/to/model.pth`.

## Predicting on the test split

Use `predict.py` to run inference on `test.csv` (or any CSV with a `smiles` column). If the CSV also contains `standardized_value`, the script will print metrics.

```bash
python predict.py \
  --data_csv /path/to/split_dir/test.csv \
  --checkpoint best_model.pth \
  --output_csv runs/exp1/test_predictions.csv
```

## Dataset

The dataset used in this project is sourced from **SweMacrocycleDB**:

## Notes

- Reproducibility controls: `--random_seed` and `--deterministic`.
- The model expects DGL graphs built from SMILES and uses optional edge features.
