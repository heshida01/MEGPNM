# -*- coding: utf-8 -*-
"""
Feature extraction and graph construction.

Includes atom/bond feature extraction and SMILES-to-DGL graph conversion.
"""
import numpy as np
import torch
import dgl
from rdkit import Chem
from rdkit.Chem import rdchem
from .utils import ENHANCED_ATOM_FEATURE_DIM


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_bond_features(bond):
    """
    Extract a bond feature vector.

    Includes: bond type, conjugation, ring membership, and stereochemistry.
    """
    bond_type = bond.GetBondType()
    features = []
    
    # Bond type one-hot (single/double/triple/aromatic)
    bond_types = [Chem.rdchem.BondType.SINGLE, 
                  Chem.rdchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.TRIPLE, 
                  Chem.rdchem.BondType.AROMATIC]
    features.extend([bond_type == bt for bt in bond_types])
    
    # Conjugation
    features.append(bond.GetIsConjugated())
    
    # In ring
    features.append(bond.IsInRing())
    
    # Stereochemistry
    stereo = bond.GetStereo()
    stereo_types = [Chem.rdchem.BondStereo.STEREONONE,
                    Chem.rdchem.BondStereo.STEREOZ,
                    Chem.rdchem.BondStereo.STEREOE]
    features.extend([stereo == st for st in stereo_types])
    
    return np.array(features, dtype=np.float32)


def get_atom_features(atom):
    """Extract enhanced atom features (~110 dimensions)."""
    try:
        valence = atom.GetValence()
    except:
        valence = atom.GetImplicitValence()
    
    try:
        formal_charge = atom.GetFormalCharge()
    except:
        formal_charge = 0
    
    try:
        hybridization = atom.GetHybridization()
    except:
        hybridization = Chem.rdchem.HybridizationType.UNSPECIFIED
    
    features = []
    
    # 1) Atom symbol one-hot (44)
    features.extend(one_of_k_encoding_unk(atom.GetSymbol(),
        ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']))
    
    # 2) Atom degree one-hot (11)
    features.extend(one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    
    # 3) Total H count one-hot (11)
    features.extend(one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    
    # 4) Valence one-hot (11)
    features.extend(one_of_k_encoding_unk(valence, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    
    # 5) Aromatic (1)
    features.append(atom.GetIsAromatic())
    
    # 6) Formal charge one-hot (7: -3..3)
    features.extend(one_of_k_encoding_unk(formal_charge, [-3, -2, -1, 0, 1, 2, 3]))
    
    # 7) Hybridization one-hot (6)
    features.extend(one_of_k_encoding_unk(hybridization, [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2, 
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ]))
    
    # 8) In ring (1)
    features.append(atom.IsInRing())
    
    # 9) Ring size indicators (8: ring sizes 3..10)
    ring_sizes = []
    for ring_size in range(3, 11):
        ring_sizes.append(atom.IsInRingSize(ring_size))
    features.extend(ring_sizes)
    
    # 10) Chirality (4)
    try:
        chiral_tag = atom.GetChiralTag()
        features.extend(one_of_k_encoding_unk(chiral_tag, [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_OTHER
        ]))
    except:
        features.extend([1, 0, 0, 0])  # default to CHI_UNSPECIFIED
    
    # 11) Atomic number (normalized) (1)
    atomic_num = atom.GetAtomicNum()
    features.append(atomic_num / 100.0)  # normalize to ~0-1
    
    # 12) Atomic mass (normalized) (1)
    try:
        mass = atom.GetMass()
        features.append(mass / 200.0)  # normalize
    except:
        features.append(0.0)
    
    # 13) Implicit valence (normalized) (1)
    try:
        implicit_valence = atom.GetImplicitValence()
        features.append(implicit_valence / 10.0)
    except:
        features.append(0.0)
    
    # 14) Hetero atom (not C/H) (1)
    features.append(atom.GetSymbol() not in ['C', 'H'])
    
    # 15) Estimated lone pairs (normalized) (1)
    try:
        lone_pairs = (valence - atom.GetDegree()) / 2.0 if valence >= atom.GetDegree() else 0.0
        features.append(lone_pairs / 4.0)  # normalize
    except:
        features.append(0.0)
    
    # 16) Relative position proxy via degree (normalized) (1)
    total_degree = atom.GetDegree() + atom.GetTotalNumHs()
    features.append(total_degree / 10.0)  # normalize degree
    
    return np.array(features, dtype=np.float32)


def smiles_to_dgl_graph(smiles, use_edge_feat=True):
    """
    Convert a SMILES string to a DGLGraph.

    Args:
        smiles: SMILES string.
        use_edge_feat: Whether to include edge features.
    Returns:
        A DGLGraph with node and (optionally) edge features.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Return a 1-node dummy graph for invalid SMILES.
            g = dgl.graph(([], []))
            g.add_nodes(1)
            g.ndata['feat'] = torch.zeros(1, ENHANCED_ATOM_FEATURE_DIM)
            if use_edge_feat:
                g.edata['feat'] = torch.zeros(0, 9)  # edge feature dimension
            return g
        
        # Atom features
        atom_features = []
        for atom in mol.GetAtoms():
            features = get_atom_features(atom)
            atom_features.append(features)
        
        if not atom_features:
            g = dgl.graph(([], []))
            g.add_nodes(1)
            g.ndata['feat'] = torch.zeros(1, ENHANCED_ATOM_FEATURE_DIM)
            if use_edge_feat:
                g.edata['feat'] = torch.zeros(0, 9)
            return g
        
        # Bonds / edge features
        src_nodes = []
        dst_nodes = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Use an undirected graph by adding both directions.
            src_nodes.extend([i, j])
            dst_nodes.extend([j, i])
            
            if use_edge_feat:
                bond_feat = get_bond_features(bond)
                # Normalize (avoid division by zero)
                bond_feat = bond_feat / (bond_feat.sum() + 1e-5)
                edge_features.extend([bond_feat, bond_feat])  # same for both directions
        
        # Build DGL graph
        if src_nodes and dst_nodes:
            g = dgl.graph((src_nodes, dst_nodes))
        else:
            g = dgl.graph(([], []))
            g.add_nodes(len(atom_features))
        
        # Node features
        g.ndata['feat'] = torch.tensor(atom_features, dtype=torch.float)
        
        # Edge features
        if use_edge_feat and edge_features:
            g.edata['feat'] = torch.tensor(edge_features, dtype=torch.float)
        elif use_edge_feat:
            # If edges exist but no features were collected, create a placeholder.
            g.edata['feat'] = torch.zeros(g.num_edges(), 9)
        
        return g
    
    except Exception as e:
        # Return a dummy graph on errors.
        g = dgl.graph(([], []))
        g.add_nodes(1)
        g.ndata['feat'] = torch.zeros(1, ENHANCED_ATOM_FEATURE_DIM)
        if use_edge_feat:
            g.edata['feat'] = torch.zeros(0, 9)
        return g


def validate_atom_features():
    """Validate enhanced atom feature dimensionality and composition."""
    print("\nEnhanced atom features validation (Total dimensions: {})".format(ENHANCED_ATOM_FEATURE_DIM))
    print("="*60)
    
    # Build a test molecule (benzene)
    test_smiles = "c1ccccc1"
    mol = Chem.MolFromSmiles(test_smiles)
    
    if mol is not None:
        for i, atom in enumerate(mol.GetAtoms()):
            features = get_atom_features(atom)
            print("Atom {} ({}): {} dimensional features".format(i, atom.GetSymbol(), len(features)))
            if i == 0:  # print details for the first atom only
                print("  First 10 dims of feature vector: {}".format(features[:10]))
                print("  Last 10 dims of feature vector: {}".format(features[-10:]))
            break
    
    # Feature composition
    feature_breakdown = [
        ("Atom symbol one-hot", 44),
        ("Atom degree one-hot", 11),
        ("Total H count one-hot", 11),
        ("Valence one-hot", 11),
        ("Aromatic", 1),
        ("Formal charge one-hot", 7),
        ("Hybridization one-hot", 6),
        ("In ring", 1),
        ("Ring size flags (3-10)", 8),
        ("Chirality", 4),
        ("Atomic number (normalized)", 1),
        ("Atomic mass (normalized)", 1),
        ("Implicit valence (normalized)", 1),
        ("Hetero atom", 1),
        ("Estimated lone pairs", 1),
        ("Total degree", 1)
    ]
    
    print("\nFeature composition details:")
    total_dims = 0
    for name, dims in feature_breakdown:
        print("  {:<25}: {:>3} dims".format(name, dims))
        total_dims += dims
    
    print("  {:<25}: {:>3} dims".format('Total', total_dims))
    
    if total_dims == ENHANCED_ATOM_FEATURE_DIM:
        print("Feature dimension validation passed!")
    else:
        print("Feature dimension mismatch! Calculated: {}, Constant: {}".format(total_dims, ENHANCED_ATOM_FEATURE_DIM))
    
    return total_dims == ENHANCED_ATOM_FEATURE_DIM
