import os
import random
import numpy as np
import torch
import dgl
import warnings
import rdkit.rdBase as rkrb

warnings.filterwarnings('ignore')
rkrb.DisableLog('rdApp.error')
rkrb.DisableLog('rdApp.warning')

ENHANCED_ATOM_FEATURE_DIM = 110


def set_random_seed(seed=42):
    print("Setting random seed to: {}".format(seed))

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dgl.random.seed(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = False

    torch.set_num_threads(1)

    print("Random seed setup complete, using deterministic algorithms")
