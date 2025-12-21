# -*- coding: utf-8 -*-
"""
Fragment-level visualization helpers.

This module provides utilities to
1) decompose molecules into BRICS fragments
2) match permeability-related SMARTS patterns
3) aggregate attention weights into fragment scores
4) render fragment heatmaps with RDKit
"""

from __future__ import annotations

import json
import os
import runpy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Set

import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS, Draw, Recap
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.Draw import rdMolDraw2D

# -----------------------------------------------------------------------------
# Data structures


@dataclass
class SmartPattern:
    """Container for SMARTS metadata."""

    name: str
    smarts: str
    description: str
    category: str
    notes: str = ""
    pattern: Optional[Chem.Mol] = field(default=None, repr=False)


@dataclass
class FragmentScore:
    """Holds fragment level statistics and metadata."""

    fragment_id: str
    atom_indices: List[int]
    base_attention: float
    attention_sum: float
    attention_max: float
    matched_smarts: List[Dict]
    final_score: float
    color_label: str
    rank: int
    fragment_type: str = "unknown"
    metadata: Dict[str, object] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# SMARTS utilities


def load_smarts_library(smarts_path: str) -> List[SmartPattern]:
    """
    Load SMARTS definitions from a python dictionary file.

    The file is expected to define PERMEABILITY_SMARTS_LIBRARY, where each entry
    contains: smarts, description, category, notes
    """
    if not os.path.exists(smarts_path):
        raise FileNotFoundError(f"SMARTS file not found: {smarts_path}")

    namespace = runpy.run_path(smarts_path)
    raw_library = namespace.get("PERMEABILITY_SMARTS_LIBRARY", {})
    patterns: List[SmartPattern] = []
    for name, payload in raw_library.items():
        smarts = payload.get("smarts")
        rd_pattern = None
        if smarts:
            rd_pattern = Chem.MolFromSmarts(smarts)
        patterns.append(
            SmartPattern(
                name=name,
                smarts=smarts or "",
                description=payload.get("description", ""),
                category=payload.get("category", ""),
                notes=payload.get("notes", ""),
                pattern=rd_pattern,
            )
        )
    return patterns


def match_smarts_patterns(
    mol: Chem.Mol, patterns: Sequence[SmartPattern]
) -> Dict[str, List[Tuple[int, ...]]]:
    """
    Return mapping from pattern name to list of matched atom index tuples.
    """
    matches: Dict[str, List[Tuple[int, ...]]] = {}
    for item in patterns:
        if not item.pattern:
            continue
        hits = mol.GetSubstructMatches(item.pattern, uniquify=True)
        if hits:
            matches[item.name] = list(hits)
    return matches


# -----------------------------------------------------------------------------
# Fragment generation


def _assign_original_indices(mol: Chem.Mol) -> None:
    """Store the original atom index before fragmentation."""
    for atom in mol.GetAtoms():
        atom.SetIntProp("orig_idx", atom.GetIdx())


def compute_brics_fragments(smiles: str) -> Tuple[Chem.Mol, List[Dict[str, object]]]:
    """
    Perform BRICS decomposition and return fragment atom index sets.

    Returns
    -------
    mol : Chem.Mol
        Sanitized RDKit molecule (without explicit hydrogens).
    fragments : List[Dict]
        Each dict contains fragment_id, type, atom_indices.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, []

    mol = Chem.RWMol(mol)
    _assign_original_indices(mol)

    bonds_to_cut: List[int] = []
    for (atom_a, atom_b), _labels in BRICS.FindBRICSBonds(mol):
        bond = mol.GetBondBetweenAtoms(atom_a, atom_b)
        if bond is not None:
            bonds_to_cut.append(bond.GetIdx())

    if bonds_to_cut:
        frag_mol = Chem.FragmentOnBonds(
            mol,
            bonds_to_cut,
            addDummies=True,
            dummyLabels=[(0, 0)] * len(bonds_to_cut),
        )
    else:
        frag_mol = mol

    fragments: List[Dict[str, object]] = []
    seen_sets = set()
    for idx, frag in enumerate(
        Chem.GetMolFrags(frag_mol, asMols=False, sanitizeFrags=False)
    ):
        atom_indices: List[int] = []
        for atom_idx in frag:
            atom = frag_mol.GetAtomWithIdx(atom_idx)
            if atom.GetAtomicNum() == 0:
                continue
            if atom.HasProp("orig_idx"):
                atom_indices.append(atom.GetIntProp("orig_idx"))
        if not atom_indices:
            continue
        atom_tuple = tuple(sorted(set(atom_indices)))
        if atom_tuple in seen_sets:
            continue
        seen_sets.add(atom_tuple)
        fragments.append(
            {
                "fragment_id": f"brics_{idx}",
                "type": "brics",
                "atom_indices": list(atom_tuple),
            }
        )

    if not fragments:
        fragments.append(
            {
                "fragment_id": "brics_full_molecule",
                "type": "brics",
                "atom_indices": list(range(mol.GetNumAtoms())),
            }
        )

    return Chem.MolFromSmiles(smiles), fragments


def compute_recap_fragments(mol: Chem.Mol) -> List[Dict[str, object]]:
    """Generate fragments using RECAP decomposition."""
    fragments: List[Dict[str, object]] = []
    if mol is None or mol.GetNumAtoms() == 0:
        return fragments
    try:
        recap_tree = Recap.RecapDecompose(mol)
    except Exception:
        return fragments
    if not recap_tree:
        return fragments

    leaves = recap_tree.GetLeaves()
    seen = set()
    counter = 0
    for path, node in leaves.items():
        leaf_mol = getattr(node, "mol", None)
        if leaf_mol is None or leaf_mol.GetNumAtoms() == 0:
            continue
        matches = mol.GetSubstructMatches(leaf_mol, uniquify=True)
        if not matches:
            continue
        leaf_smiles = Chem.MolToSmiles(leaf_mol, canonical=True)
        for match in matches:
            atom_indices = tuple(sorted(set(int(idx) for idx in match)))
            if not atom_indices or atom_indices in seen:
                continue
            seen.add(atom_indices)
            fragments.append(
                {
                    "fragment_id": f"recap_{counter}",
                    "type": "recap",
                    "atom_indices": list(atom_indices),
                    "metadata": {
                        "recap_path": str(path),
                        "recap_smiles": leaf_smiles,
                    },
                }
            )
            counter += 1
    return fragments


def compute_murcko_scaffold_fragments(mol: Chem.Mol) -> List[Dict[str, object]]:
    """Extract Bemis-Murcko scaffolds as fragments."""
    fragments: List[Dict[str, object]] = []
    if mol is None or mol.GetNumAtoms() == 0:
        return fragments
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return fragments
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return fragments

    scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)
    matches = mol.GetSubstructMatches(scaffold, uniquify=True)
    seen = set()
    for idx, match in enumerate(matches):
        atom_indices = tuple(sorted(int(i) for i in match))
        if not atom_indices or atom_indices in seen:
            continue
        seen.add(atom_indices)
        fragments.append(
            {
                "fragment_id": f"murcko_{idx}",
                "type": "murcko",
                "atom_indices": list(atom_indices),
                "metadata": {"scaffold_smiles": scaffold_smiles},
            }
        )
    return fragments


def compute_macrocycle_fragments(
    mol: Chem.Mol, min_size: int = 12
) -> List[Dict[str, object]]:
    """Identify macrocyclic rings (>= min_size atoms)."""
    fragments: List[Dict[str, object]] = []
    if mol is None or mol.GetNumAtoms() == 0:
        return fragments
    try:
        ring_info = mol.GetRingInfo()
    except Exception:
        return fragments
    seen = set()
    counter = 0
    for ring in ring_info.AtomRings():
        ring_size = len(ring)
        if ring_size < min_size:
            continue
        atom_indices = tuple(sorted(int(i) for i in ring))
        if atom_indices in seen:
            continue
        seen.add(atom_indices)
        fragments.append(
            {
                "fragment_id": f"macrocycle_{counter}",
                "type": "macrocycle",
                "atom_indices": list(atom_indices),
                "metadata": {"ring_size": ring_size},
            }
        )
        counter += 1
    return fragments


def build_smarts_fragments(
    smarts_matches: Dict[str, List[Tuple[int, ...]]]
) -> List[Dict[str, object]]:
    """
    Convert SMARTS matches into fragment definitions.
    """
    fragments: List[Dict[str, object]] = []
    seen = set()
    for pattern_name, hits in smarts_matches.items():
        for match_idx, atoms in enumerate(hits):
            atom_indices = sorted(set(int(idx) for idx in atoms))
            if not atom_indices:
                continue
            key = (pattern_name, tuple(atom_indices))
            if key in seen:
                continue
            seen.add(key)
            fragments.append(
                {
                    "fragment_id": f"smarts_{pattern_name}_{match_idx}",
                    "type": "smarts",
                    "atom_indices": atom_indices,
                    "metadata": {
                        "smarts_name": pattern_name,
                        "match_index": match_idx,
                    },
                }
            )
    return fragments


# -----------------------------------------------------------------------------
# Scoring helpers


def aggregate_fragment_scores(
    fragments: Sequence[Dict[str, object]],
    smarts_matches: Dict[str, List[Tuple[int, ...]]],
    smarts_patterns: Sequence[SmartPattern],
    attention: np.ndarray,
    impact_weight: float = 0.1,
    color_palette: Optional[Dict[str, Tuple[float, float, float]]] = None,
    topk: int = 3,
    min_atoms: int = 1,
    max_atoms: Optional[int] = None,
) -> List[FragmentScore]:
    """
    Combine attention statistics with SMARTS metadata and produce ranked scores.
    """
    if color_palette is None:
        color_palette = {
            "top1": (1.00, 0.38, 0.50),  # vivid rose pink
            "top2": (1.00, 0.62, 0.26),  # vibrant tangerine
            "top3": (0.32, 0.90, 0.66),  # bright turquoise
            "others": (0.80, 0.80, 0.80),  # light grey
        }

    def _lookup_pattern(name: str) -> Optional[SmartPattern]:
        for pattern in smarts_patterns:
            if pattern.name == name:
                return pattern
        return None

    attention = np.asarray(attention).reshape(-1)
    eligible: List[FragmentScore] = []
    all_scores: List[FragmentScore] = []
    seen_atom_sets: Set[Tuple[int, ...]] = set()
    for frag in fragments:
        atom_indices = sorted(int(idx) for idx in frag.get("atom_indices", []))
        if not atom_indices:
            continue
        atom_indices = [idx for idx in atom_indices if idx < len(attention)]
        if not atom_indices:
            continue
        atom_tuple = tuple(atom_indices)
        if atom_tuple in seen_atom_sets:
            continue
        seen_atom_sets.add(atom_tuple)
        atom_count = len(atom_indices)
        attention_values = attention[atom_indices]
        base_attention = float(attention_values.mean())
        attention_sum = float(attention_values.sum())
        attention_max = float(attention_values.max())

        matched_names: List[str] = []
        matched_details: List[Dict] = []
        for name, hit_list in smarts_matches.items():
            for hit in hit_list:
                if any(idx in atom_indices for idx in hit):
                    matched_names.append(name)
                    pattern = _lookup_pattern(name)
                    matched_details.append(
                        {
                            "name": name,
                            "atoms": list(hit),
                            "category": pattern.category if pattern else "",
                            "description": pattern.description if pattern else "",
                        }
                    )
        final_score = base_attention

        frag_score = FragmentScore(
            fragment_id=str(frag.get("fragment_id")),
            atom_indices=atom_indices,
            base_attention=base_attention,
            attention_sum=attention_sum,
            attention_max=attention_max,
            matched_smarts=matched_details,
            final_score=float(final_score),
            color_label="others",
            rank=-1,
            fragment_type=str(frag.get("type", "unknown")),
            metadata=dict(frag.get("metadata", {})),
        )
        if atom_count < max(min_atoms, 1):
            frag_score.warnings.append(
                f"Fragment skipped from ranking (size {atom_count} < min_atoms {min_atoms})"
            )
            all_scores.append(frag_score)
            continue
        if max_atoms is not None and atom_count > max_atoms:
            frag_score.warnings.append(
                f"Fragment skipped from ranking (size {atom_count} > max_atoms {max_atoms})"
            )
            all_scores.append(frag_score)
            continue

        eligible.append(frag_score)
        all_scores.append(frag_score)

    eligible.sort(key=lambda x: x.final_score, reverse=True)

    label_order = ["top1", "top2", "top3"]
    for rank, item in enumerate(eligible):
        if rank < topk and rank < len(label_order):
            item.color_label = label_order[rank]
        else:
            item.color_label = "others"
        item.rank = rank + 1
    # Ensure non-eligible fragments remain with default rank/color

    all_scores.sort(key=lambda x: x.final_score, reverse=True)
    return all_scores


# -----------------------------------------------------------------------------
# Visualization helpers


def build_highlight_maps(
    mol: Chem.Mol, fragments: Sequence[FragmentScore], palette: Dict[str, Tuple[float, float, float]]
) -> Tuple[List[int], Dict[int, Tuple[float, float, float]], Dict[int, float], List[int], Dict[int, Tuple[float, float, float]]]:
    """
    Prepare highlight dictionaries for RDKit Draw API.
    """
    atom_colors: Dict[int, Tuple[float, float, float]] = {}
    atom_radii: Dict[int, float] = {}
    atom_priority: Dict[int, int] = {}
    bond_colors: Dict[int, Tuple[float, float, float]] = {}
    highlight_atoms: List[int] = []
    highlight_bonds: List[int] = []

    # Adjust priority so top3 fragments can override broader top2 regions,
    # while top1 remains the strongest highlight.
    priority_map = {"top1": 0, "top3": 1, "top2": 2, "others": 3}

    # Identify overlaps among Top-K fragments
    atom_top_counts = {}
    top_labels = {"top1", "top2", "top3"}
    for frag in fragments:
        if frag.color_label in top_labels:
            for atom_idx in frag.atom_indices:
                if atom_idx < mol.GetNumAtoms():
                    atom_top_counts[atom_idx] = atom_top_counts.get(atom_idx, 0) + 1

    overlap_color = (0.0, 0.0, 1.0)  # Blue for overlaps
    overlap_priority = -1

    for frag in fragments:
        color = palette.get(frag.color_label, palette.get("others", (0.7, 0.7, 0.7)))
        priority = priority_map.get(frag.color_label, 3)
        radius = 0.55 if priority <= 2 else 0.35

        for atom_idx in frag.atom_indices:
            if atom_idx >= mol.GetNumAtoms():
                continue

            # Check for overlap
            is_overlap = atom_top_counts.get(atom_idx, 0) > 1

            current_color = overlap_color if is_overlap else color
            current_priority = overlap_priority if is_overlap else priority
            current_radius = 0.55 if is_overlap else radius

            if atom_idx not in atom_priority or current_priority < atom_priority[atom_idx]:
                atom_colors[atom_idx] = current_color
                atom_radii[atom_idx] = current_radius
                atom_priority[atom_idx] = current_priority

        highlight_atoms.extend(
            [idx for idx in frag.atom_indices if idx < mol.GetNumAtoms()]
        )

    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in atom_colors and end in atom_colors:
            bond_idx = bond.GetIdx()
            begin_priority = atom_priority.get(begin, 3)
            end_priority = atom_priority.get(end, 3)
            chosen_atom = begin if begin_priority <= end_priority else end
            bond_colors[bond_idx] = atom_colors[chosen_atom]
            highlight_bonds.append(bond_idx)

    highlight_atoms = sorted(set(highlight_atoms))
    highlight_bonds = sorted(set(highlight_bonds))

    if fragments:
        max_score = max(frag.final_score for frag in fragments if frag.final_score is not None)
    else:
        max_score = 1.0
    max_score = max(max_score, 1e-6)

    for frag in fragments:
        if frag.color_label not in {"top1", "top2", "top3"}:
            continue
        score_ratio = max(0.0, min(1.0, float(frag.final_score) / max_score))
        base_color = palette.get(frag.color_label, palette.get("others", (0.5, 0.5, 0.5)))
        scaling = 0.5 + 0.5 * score_ratio
        adjusted_color = tuple(max(0.0, min(1.0, scaling * c)) for c in base_color)

        for atom_idx in frag.atom_indices:
            if atom_idx >= mol.GetNumAtoms():
                continue
            # Skip if it is an overlap (blue)
            if atom_colors.get(atom_idx) == overlap_color:
                continue
            if atom_colors.get(atom_idx) == base_color:
                atom_colors[atom_idx] = adjusted_color

        for bond in mol.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            if begin in frag.atom_indices and end in frag.atom_indices:
                bond_idx = bond.GetIdx()
                if bond_colors.get(bond_idx) == base_color:
                    bond_colors[bond_idx] = adjusted_color

    return highlight_atoms, atom_colors, atom_radii, highlight_bonds, bond_colors


def render_fragment_visualization(
    smiles: str,
    fragments: Sequence[FragmentScore],
    output_path: str,
    palette: Optional[Dict[str, Tuple[float, float, float]]] = None,
    size: Tuple[int, int] = (520, 420),
) -> bool:
    """
    Render highlighted molecule with fragment colors.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    if palette is None:
        palette = {
            "top1": (1.00, 0.38, 0.50),  # vivid rose pink
            "top2": (1.00, 0.62, 0.26),  # vibrant tangerine
            "top3": (0.32, 0.90, 0.66),  # bright turquoise
            "others": (0.80, 0.80, 0.80),  # light grey
        }

    highlight_atoms, atom_colors, atom_radii, highlight_bonds, bond_colors = (
        build_highlight_maps(mol, fragments, palette)
    )

    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer,
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=atom_colors,
            highlightAtomRadii=atom_radii,
            highlightBonds=highlight_bonds,
            highlightBondColors=bond_colors,
        )
        drawer.FinishDrawing()
        with open(output_path, "wb") as f:
            f.write(drawer.GetDrawingText())
        return True
    except Exception:
        try:
            img = Draw.MolToImage(mol, size=size)
            img.save(output_path)
            return True
        except Exception:
            return False


# -----------------------------------------------------------------------------
# Serialization helpers


def fragment_scores_to_json(
    sample_idx: int,
    smiles: str,
    prediction: float,
    fragments: Sequence[FragmentScore],
    output_path: str,
    extra_info: Optional[Dict[str, object]] = None,
) -> None:
    """
    Persist per-molecule fragment details to JSON.
    """
    payload = {
        "sample_index": sample_idx,
        "smiles": smiles,
        "prediction": prediction,
        "fragments": [
            {
                "fragment_id": frag.fragment_id,
                "atom_indices": frag.atom_indices,
                "rank": frag.rank,
                "color": frag.color_label,
                "fragment_type": frag.fragment_type,
                "metadata": frag.metadata,
                "scores": {
                    "attention_mean": frag.base_attention,
                    "attention_sum": frag.attention_sum,
                    "attention_max": frag.attention_max,
                    "final_score": frag.final_score,
                },
                "smart_matches": frag.matched_smarts,
                "warnings": frag.warnings,
            }
            for frag in fragments
        ],
    }
    if extra_info:
        payload.update(extra_info)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def ensure_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
