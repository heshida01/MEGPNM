# MEGPNM

**Multiscale edge-aware graph attention network with hybrid pooling predicts membrane permeability of non-peptidic macrocycles**

## Installation

```bash
conda create -n megpnm python=3.11 && conda activate megpnm

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
pip install -r requirements.txt
```

Reported results used Python 3.11.13, PyTorch 2.4.0+cu121, DGL 2.4.0+cu121,
RDKit 2025.09.1, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, SciPy 1.16.0.

## Data

Non-peptidic macrocycles with PAMPA permeability values, derived from
[SweMacrocycleDB](https://swemacrocycledb.com/).

| Directory | Split |
| --- | --- |
| `dataset/random/` | Random |
| `dataset/cliff/` | Activity cliff |
| `dataset/scaffold/` | Bemis–Murcko scaffold |

Columns: `smiles`, and `standardized_value` (log *P*<sub>app</sub>, roughly −8 to −4.3).

## Training

```bash
python train.py --data_dir dataset/random --save_dir runs/exp1 \
                --epochs 180 --random_seed 42
```

**Checkpoint selection is validation-only.** The checkpoint with the lowest
validation RMSE is kept; the test set is scored exactly once at the end using that
checkpoint and never influences training or model selection.

Useful flags: `--device`, `--no_save_model`, `--skip_train` (export predictions from
an existing `--checkpoint`).

## Prediction

```bash
python predict.py --data_csv dataset/random/test.csv \
                  --checkpoint runs/exp1/best_model.pth \
                  --output_csv runs/exp1/test_predictions.csv
```

## Citation

Please cite the MEGPNM paper if you use this code, and also cite [SweMacrocycleDB](https://swemacrocycledb.com/) if you use the data.