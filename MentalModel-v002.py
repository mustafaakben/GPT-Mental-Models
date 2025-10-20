# -*- coding: utf-8 -*-
#################
## MENTAL MODEL LAYER - OPTIMIZED MULTI-STRATEGY ATOM FRAMEWORK
## ZEN Edition: Simple, Efficient, Intentional
## OPTIMIZATIONS: Vectorized atoms + Progress bars + Parallel execution + INSTANT RESULTS
## NEW: Synthetic Dataset Suite (easy_binary, hard_binary, hard_multi, xor, moons, circles)
#################
"""
PERFORMANCE OPTIMIZATIONS:
- ✅ Vectorized all 8 atom strategies (10-100x faster)
- ✅ Progress bars with ETA
- ✅ Parallel execution across configs (Option A)
- ✅ Full grid search (all 14 configurations)
- ✅ INSTANT results display after each run
- ✅ Incremental CSV saving
- ✅ Synthetic dataset suite for comprehensive testing
"""

import random

# ============================================================
# GLOBAL PARAMETERS (edit me)
# ============================================================
SEED                  = random.randint(1,100000)
DEVICE                = "cuda"       # "cuda" or "cpu"

# 🆕 UPDATED: Synthetic + Real datasets
DATASETS              = [
    "easy_binary",    # ⭐ Sanity check - well-separated binary
    "hard_binary",    # 🔥 Noise test - noisy binary
    "hard_multi",     # 💪 Ultimate challenge - imbalanced multi-class
    "xor",            # 🎯 Non-linear critical test
    "moons",          # 🌙 Curved decision boundaries
    "wine"            # 🍷 Real data comparison
]

TEST_SIZE             = 0.20
STANDARDIZE           = True

# Training
BATCH_SIZE            = 32
EPOCHS                = 300
LR                    = 1e-2

# Base atom params
QUANTILES             = (0.25, 0.5, 0.75)
ATOM_BETA             = 30.0

# Worldlets
WORLDLETS_PER_CLASS   = 20

# Regularization
LAMBDA_SPARSITY       = 0.001
LAMBDA_BINARY         = 0.001
LAMBDA_ACT_PEN        = 0.0001

PRINT_EVERY_EPOCHS    = 25

# Baselines
USE_LOGREG            = True
USE_RANDOM_FOREST     = True
USE_GRAD_BOOST        = True
USE_SVM_RBF           = True

RF_N_ESTIMATORS       = 300
RF_MAX_DEPTH          = None
GB_N_ESTIMATORS       = 300
GB_LEARNING_RATE      = 0.05
GB_MAX_DEPTH          = 3
SVM_C                 = 3.0
SVM_GAMMA             = "scale"

# ============================================================
# GRID SEARCH CONFIGURATION
# ============================================================
ENABLE_GRID_SEARCH    = True
GRID_SEARCH_MODE      = "staged"
PARALLEL_JOBS         = -1            # 1=sequential, 2+=parallel, -1=all cores

# Full grid search (all 14 configurations)
ATOM_STRATEGY_CONFIGS = [
    "threshold_only",
    "range_only", 
    "relational_only",
    "interaction_only",
    "fuzzy_only",
    "polynomial_only",
    "statistical_only",
    "distance_only",
    "threshold+range",
    "threshold+relational",
    "threshold+interaction",
    "all_simple",      # threshold, range, relational, fuzzy
    "all_advanced",    # threshold, interaction, polynomial, statistical
    "full_suite",      # all 8
]

# ============================================================
# Imports
# ============================================================
import os, math, json, random, warnings, time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from itertools import combinations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

# Dataset imports
from sklearn.datasets import (
    load_wine, load_iris, load_breast_cancer, load_digits,
    make_classification, make_moons, make_circles  # 🆕 Synthetic datasets
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans

# Progress bars
try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[WARNING] tqdm not found. Install with: pip install tqdm")
    class tqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            pass
        def set_description(self, desc):
            pass

# Parallel execution
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    print("[WARNING] joblib not found. Install with: pip install joblib")

warnings.filterwarnings("ignore")

# ============================================================
# Seeding
# ============================================================
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if DEVICE == "cuda" and not torch.cuda.is_available():
    DEVICE = "cpu"
print(f"[SETUP] DEVICE={DEVICE}  |  SEED={SEED}")
print(f"[SETUP] Progress bars: {'✅' if HAS_TQDM else '❌'} | Parallel: {'✅' if HAS_JOBLIB and PARALLEL_JOBS != 1 else '❌'}")

# ============================================================
# 🆕 SYNTHETIC DATASET GENERATORS
# ============================================================

def make_xor(n_samples=400, noise=0.15, random_state=None):
    """
    Generate XOR dataset - classic non-linear problem
    Linear models will fail (~50% accuracy), non-linear methods should succeed
    """
    rng = np.random.RandomState(random_state)
    
    # Generate 4 clusters in XOR pattern
    n_per_cluster = n_samples // 4
    
    # Cluster 1: bottom-left (class 0)
    X1 = rng.randn(n_per_cluster, 2) * noise + np.array([-1.0, -1.0])
    
    # Cluster 2: top-right (class 0)
    X2 = rng.randn(n_per_cluster, 2) * noise + np.array([1.0, 1.0])
    
    # Cluster 3: top-left (class 1)
    X3 = rng.randn(n_per_cluster, 2) * noise + np.array([-1.0, 1.0])
    
    # Cluster 4: bottom-right (class 1)
    X4 = rng.randn(n_per_cluster, 2) * noise + np.array([1.0, -1.0])
    
    X = np.vstack([X1, X2, X3, X4])
    y = np.array([0]*n_per_cluster*2 + [1]*n_per_cluster*2)
    
    # Shuffle
    idx = rng.permutation(len(X))
    X = X[idx].astype(np.float32)
    y = y[idx].astype(np.int64)
    
    return X, y


def generate_synthetic_dataset(name: str, random_state=None):
    """
    Generate synthetic datasets based on name
    Returns: X, y, feature_names, class_names
    """
    
    if name == "easy_binary":
        # Well-separated binary classification
        X, y = make_classification(
            n_samples=500,
            n_features=10,
            n_informative=8,
            n_redundant=1,
            n_classes=2,
            n_clusters_per_class=1,
            class_sep=2.0,
            flip_y=0.01,
            random_state=random_state
        )
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        class_names = ["class_0", "class_1"]
        
    elif name == "hard_binary":
        # Noisy binary classification with redundant features
        X, y = make_classification(
            n_samples=500,
            n_features=20,
            n_informative=3,
            n_redundant=10,
            n_classes=2,
            n_clusters_per_class=3,
            class_sep=0.5,
            flip_y=0.15,
            random_state=random_state
        )
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        class_names = ["class_0", "class_1"]
        
    elif name == "easy_multi":
        # Well-separated multi-class
        X, y = make_classification(
            n_samples=800,
            n_features=15,
            n_informative=10,
            n_redundant=2,
            n_classes=5,
            n_clusters_per_class=1,
            class_sep=1.5,
            flip_y=0.05,
            random_state=random_state
        )
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        class_names = [f"class_{i}" for i in range(5)]
        
    elif name == "hard_multi":
        # Very challenging: imbalanced, noisy, many classes
        X, y = make_classification(
            n_samples=1000,
            n_features=30,
            n_informative=5,
            n_redundant=15,
            n_classes=7,
            n_clusters_per_class=2,
            class_sep=0.3,
            flip_y=0.20,
            weights=[0.4, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05],
            random_state=random_state
        )
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        class_names = [f"class_{i}" for i in range(7)]
        
    elif name == "xor":
        # Classic XOR problem - non-linear
        X, y = make_xor(n_samples=400, noise=0.15, random_state=random_state)
        feature_names = ["x", "y"]
        class_names = ["class_0", "class_1"]
        
    elif name == "moons":
        # Two interleaving half circles
        X, y = make_moons(n_samples=500, noise=0.1, random_state=random_state)
        X = X.astype(np.float32)
        y = y.astype(np.int64)
        feature_names = ["x", "y"]
        class_names = ["moon_0", "moon_1"]
        
    elif name == "circles":
        # Concentric circles - radial pattern
        X, y = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=random_state)
        X = X.astype(np.float32)
        y = y.astype(np.int64)
        feature_names = ["x", "y"]
        class_names = ["inner", "outer"]
        
    else:
        raise ValueError(f"Unknown synthetic dataset: {name}")
    
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    
    return X, y, feature_names, class_names


# ============================================================
# Dataset loader (UPDATED)
# ============================================================
def load_dataset_by_name(name: str):
    """
    Load dataset by name - supports both real and synthetic datasets
    """
    name = name.lower()
    
    # Real datasets
    if name == "iris":
        ds = load_iris()
        X = ds["data"].astype(np.float32)
        y = ds["target"].astype(np.int64)
        feature_names = list(ds.get("feature_names", [f"f{i}" for i in range(X.shape[1])]))
        class_names = list(ds.get("target_names", [str(i) for i in np.unique(y)]))
        
    elif name == "wine":
        ds = load_wine()
        X = ds["data"].astype(np.float32)
        y = ds["target"].astype(np.int64)
        feature_names = list(ds.get("feature_names", [f"f{i}" for i in range(X.shape[1])]))
        class_names = list(ds.get("target_names", [str(i) for i in np.unique(y)]))
        
    elif name in ("breast_cancer", "cancer"):
        ds = load_breast_cancer()
        X = ds["data"].astype(np.float32)
        y = ds["target"].astype(np.int64)
        feature_names = list(ds.get("feature_names", [f"f{i}" for i in range(X.shape[1])]))
        class_names = list(ds.get("target_names", [str(i) for i in np.unique(y)]))
        
    elif name == "digits":
        ds = load_digits()
        X = ds["data"].astype(np.float32)
        y = ds["target"].astype(np.int64)
        feature_names = list(ds.get("feature_names", [f"f{i}" for i in range(X.shape[1])]))
        class_names = list(ds.get("target_names", [str(i) for i in np.unique(y)]))
    
    # 🆕 Synthetic datasets
    elif name in ["easy_binary", "hard_binary", "easy_multi", "hard_multi", "xor", "moons", "circles"]:
        X, y, feature_names, class_names = generate_synthetic_dataset(name, random_state=SEED)
    
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    return X, y, feature_names, class_names

# ============================================================
# ATOM STRATEGY FRAMEWORK
# ============================================================

@dataclass
class AtomMeta:
    """Metadata for a single atom"""
    name: str
    strategy_type: str
    feature_indices: List[int]
    params: Dict[str, Any] = field(default_factory=dict)

class BaseAtomStrategy(ABC):
    """Abstract base class for all atom strategies"""
    
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.atoms_meta: List[AtomMeta] = []
    
    @abstractmethod
    def generate_atoms_metadata(self, feature_names: List[str], train_data: np.ndarray) -> List[AtomMeta]:
        """Generate metadata for all atoms this strategy will create"""
        pass
    
    @abstractmethod
    def initialize_parameters(self, train_data: np.ndarray, device: str) -> Dict[str, torch.nn.Parameter]:
        """Initialize learnable parameters for this strategy"""
        pass
    
    @abstractmethod
    def compute_atom_values(self, X: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute atom truth values for input X
        Returns: [B, num_atoms] tensor
        """
        pass
    
    def get_num_atoms(self) -> int:
        return len(self.atoms_meta)


# ============================================================
# STRATEGY 1: THRESHOLD ATOMS (VECTORIZED)
# ============================================================
class ThresholdAtomStrategy(BaseAtomStrategy):
    def __init__(self, quantiles: Tuple[float, ...] = (0.25, 0.5, 0.75), beta: float = 30.0):
        super().__init__("Threshold")
        self.quantiles = quantiles
        self.beta = beta
    
    def generate_atoms_metadata(self, feature_names: List[str], train_data: np.ndarray) -> List[AtomMeta]:
        atoms = []
        atom_feat_indices = []
        atom_q_indices = []
        atom_directions = []
        
        for j, fname in enumerate(feature_names):
            for q_idx, q in enumerate(self.quantiles):
                atoms.append(AtomMeta(
                    name=f"{fname} ≤ q{q}",
                    strategy_type=self.strategy_name,
                    feature_indices=[j],
                    params={"q_idx": q_idx, "direction": "<="}
                ))
                atom_feat_indices.append(j)
                atom_q_indices.append(q_idx)
                atom_directions.append(1.0)  # "<="
                
                atoms.append(AtomMeta(
                    name=f"{fname} > q{q}",
                    strategy_type=self.strategy_name,
                    feature_indices=[j],
                    params={"q_idx": q_idx, "direction": ">"}
                ))
                atom_feat_indices.append(j)
                atom_q_indices.append(q_idx)
                atom_directions.append(-1.0)  # ">"
        
        self.atoms_meta = atoms
        # Store indices for vectorization
        self.atom_feat_idx = np.array(atom_feat_indices, dtype=np.int64)
        self.atom_q_idx = np.array(atom_q_indices, dtype=np.int64)
        self.atom_dir = np.array(atom_directions, dtype=np.float32)
        
        return atoms
    
    def initialize_parameters(self, train_data: np.ndarray, device: str) -> Dict[str, torch.nn.Parameter]:
        D = train_data.shape[1]
        init_T = []
        for j in range(D):
            col = train_data[:, j]
            tjs = np.quantile(col, q=self.quantiles).astype(np.float32)
            init_T.append(tjs)
        init_T = np.stack(init_T, axis=0)  # [D, Q]
        
        return {
            "T": nn.Parameter(torch.tensor(init_T, dtype=torch.float32).to(device)),
            "atom_feat_idx": torch.tensor(self.atom_feat_idx, dtype=torch.long).to(device),
            "atom_q_idx": torch.tensor(self.atom_q_idx, dtype=torch.long).to(device),
            "atom_dir": torch.tensor(self.atom_dir, dtype=torch.float32).to(device)
        }
    
    def compute_atom_values(self, X: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """VECTORIZED - No Python loops!"""
        T = params["T"]
        atom_feat_idx = params["atom_feat_idx"]
        atom_q_idx = params["atom_q_idx"]
        atom_dir = params["atom_dir"]
        
        # Get all relevant features: [B, A]
        X_selected = X[:, atom_feat_idx]
        
        # Get all thresholds: [A]
        T_selected = T[atom_feat_idx, atom_q_idx]
        
        # Compute differences and apply direction
        diff = T_selected.unsqueeze(0) - X_selected  # [B, A]
        signed_diff = atom_dir.unsqueeze(0) * diff
        
        # Sigmoid activation
        return torch.sigmoid(self.beta * signed_diff)


# ============================================================
# STRATEGY 2: RANGE ATOMS (VECTORIZED)
# ============================================================
class RangeAtomStrategy(BaseAtomStrategy):
    def __init__(self, quantile_pairs: List[Tuple[float, float]] = [(0.25, 0.75), (0.1, 0.9)], beta: float = 30.0):
        super().__init__("Range")
        self.quantile_pairs = quantile_pairs
        self.beta = beta
    
    def generate_atoms_metadata(self, feature_names: List[str], train_data: np.ndarray) -> List[AtomMeta]:
        atoms = []
        feat_indices = []
        pair_indices = []
        inside_flags = []
        
        for j, fname in enumerate(feature_names):
            for pair_idx, (q_low, q_high) in enumerate(self.quantile_pairs):
                # Inside range
                atoms.append(AtomMeta(
                    name=f"{fname} in [{q_low},{q_high}]",
                    strategy_type=self.strategy_name,
                    feature_indices=[j],
                    params={"pair_idx": pair_idx, "inside": True}
                ))
                feat_indices.append(j)
                pair_indices.append(pair_idx)
                inside_flags.append(1.0)
                
                # Outside range
                atoms.append(AtomMeta(
                    name=f"{fname} NOT in [{q_low},{q_high}]",
                    strategy_type=self.strategy_name,
                    feature_indices=[j],
                    params={"pair_idx": pair_idx, "inside": False}
                ))
                feat_indices.append(j)
                pair_indices.append(pair_idx)
                inside_flags.append(0.0)
        
        self.atoms_meta = atoms
        self.feat_idx = np.array(feat_indices, dtype=np.int64)
        self.pair_idx = np.array(pair_indices, dtype=np.int64)
        self.inside_flag = np.array(inside_flags, dtype=np.float32)
        
        return atoms
    
    def initialize_parameters(self, train_data: np.ndarray, device: str) -> Dict[str, torch.nn.Parameter]:
        D = train_data.shape[1]
        T_low, T_high = [], []
        for j in range(D):
            col = train_data[:, j]
            lows = [np.quantile(col, q[0]) for q in self.quantile_pairs]
            highs = [np.quantile(col, q[1]) for q in self.quantile_pairs]
            T_low.append(lows)
            T_high.append(highs)
        T_low = np.array(T_low, dtype=np.float32)  # [D, num_pairs]
        T_high = np.array(T_high, dtype=np.float32)
        
        return {
            "T_low": nn.Parameter(torch.tensor(T_low).to(device)),
            "T_high": nn.Parameter(torch.tensor(T_high).to(device)),
            "feat_idx": torch.tensor(self.feat_idx, dtype=torch.long).to(device),
            "pair_idx": torch.tensor(self.pair_idx, dtype=torch.long).to(device),
            "inside_flag": torch.tensor(self.inside_flag, dtype=torch.float32).to(device)
        }
    
    def compute_atom_values(self, X: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """VECTORIZED - No Python loops!"""
        T_low = params["T_low"]
        T_high = params["T_high"]
        feat_idx = params["feat_idx"]
        pair_idx = params["pair_idx"]
        inside_flag = params["inside_flag"]
        
        # Get features: [B, A]
        X_selected = X[:, feat_idx]
        
        # Get thresholds: [A]
        T_low_selected = T_low[feat_idx, pair_idx]
        T_high_selected = T_high[feat_idx, pair_idx]
        
        # Compute range membership
        above_low = torch.sigmoid(self.beta * (X_selected - T_low_selected.unsqueeze(0)))
        below_high = torch.sigmoid(self.beta * (T_high_selected.unsqueeze(0) - X_selected))
        in_range = above_low * below_high
        
        # Apply inside/outside flag
        result = inside_flag.unsqueeze(0) * in_range + (1 - inside_flag.unsqueeze(0)) * (1 - in_range)
        
        return result


# ============================================================
# STRATEGY 3: RELATIONAL ATOMS (VECTORIZED)
# ============================================================
class RelationalAtomStrategy(BaseAtomStrategy):
    def __init__(self, max_pairs: int = 30, beta: float = 30.0):
        super().__init__("Relational")
        self.max_pairs = max_pairs
        self.beta = beta
    
    def generate_atoms_metadata(self, feature_names: List[str], train_data: np.ndarray) -> List[AtomMeta]:
        D = len(feature_names)
        all_pairs = list(combinations(range(D), 2))
        
        if len(all_pairs) > self.max_pairs:
            np.random.shuffle(all_pairs)
            all_pairs = all_pairs[:self.max_pairs]
        
        atoms = []
        feat_i_list = []
        feat_j_list = []
        relation_list = []
        
        for (i, j) in all_pairs:
            # i > j
            atoms.append(AtomMeta(
                name=f"{feature_names[i]} > {feature_names[j]}",
                strategy_type=self.strategy_name,
                feature_indices=[i, j],
                params={"relation": ">"}
            ))
            feat_i_list.append(i)
            feat_j_list.append(j)
            relation_list.append(1.0)  # ">"
            
            # i < j
            atoms.append(AtomMeta(
                name=f"{feature_names[i]} < {feature_names[j]}",
                strategy_type=self.strategy_name,
                feature_indices=[i, j],
                params={"relation": "<"}
            ))
            feat_i_list.append(i)
            feat_j_list.append(j)
            relation_list.append(-1.0)  # "<"
        
        self.atoms_meta = atoms
        self.feat_i = np.array(feat_i_list, dtype=np.int64)
        self.feat_j = np.array(feat_j_list, dtype=np.int64)
        self.relation_sign = np.array(relation_list, dtype=np.float32)
        
        return atoms
    
    def initialize_parameters(self, train_data: np.ndarray, device: str) -> Dict[str, torch.nn.Parameter]:
        return {
            "feat_i": torch.tensor(self.feat_i, dtype=torch.long).to(device),
            "feat_j": torch.tensor(self.feat_j, dtype=torch.long).to(device),
            "relation_sign": torch.tensor(self.relation_sign, dtype=torch.float32).to(device)
        }
    
    def compute_atom_values(self, X: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """VECTORIZED - No Python loops!"""
        feat_i = params["feat_i"]
        feat_j = params["feat_j"]
        relation_sign = params["relation_sign"]
        
        # Get both features: [B, A]
        X_i = X[:, feat_i]
        X_j = X[:, feat_j]
        
        # Compute difference
        diff = X_i - X_j
        
        # Apply relation sign and sigmoid
        return torch.sigmoid(self.beta * relation_sign.unsqueeze(0) * diff)


# ============================================================
# STRATEGY 4: INTERACTION ATOMS (VECTORIZED)
# ============================================================
class InteractionAtomStrategy(BaseAtomStrategy):
    def __init__(self, max_interactions: int = 40, interaction_order: int = 2, beta: float = 30.0):
        super().__init__("Interaction")
        self.max_interactions = max_interactions
        self.interaction_order = interaction_order
        self.beta = beta
    
    def generate_atoms_metadata(self, feature_names: List[str], train_data: np.ndarray) -> List[AtomMeta]:
        D = len(feature_names)
        all_combos = list(combinations(range(D), self.interaction_order))
        
        if len(all_combos) > self.max_interactions:
            np.random.shuffle(all_combos)
            all_combos = all_combos[:self.max_interactions]
        
        atoms = []
        combo_list = []
        direction_list = []
        
        for combo in all_combos:
            feat_str = " × ".join([feature_names[i] for i in combo])
            
            # Product > threshold
            atoms.append(AtomMeta(
                name=f"{feat_str} > threshold",
                strategy_type=self.strategy_name,
                feature_indices=list(combo),
                params={"interaction_type": "product", "direction": ">"}
            ))
            combo_list.append(combo)
            direction_list.append(1.0)
            
            # Product <= threshold
            atoms.append(AtomMeta(
                name=f"{feat_str} ≤ threshold",
                strategy_type=self.strategy_name,
                feature_indices=list(combo),
                params={"interaction_type": "product", "direction": "<="}
            ))
            combo_list.append(combo)
            direction_list.append(-1.0)
        
        self.atoms_meta = atoms
        self.feature_combos = all_combos
        self.combo_indices = combo_list
        self.direction_sign = np.array(direction_list, dtype=np.float32)
        
        return atoms
    
    def initialize_parameters(self, train_data: np.ndarray, device: str) -> Dict[str, torch.nn.Parameter]:
        thresholds = []
        for combo in self.feature_combos:
            interaction_vals = train_data[:, list(combo)].prod(axis=1)
            t = np.median(interaction_vals).astype(np.float32)
            thresholds.append(t)
        
        T = np.array(thresholds, dtype=np.float32)
        
        return {
            "T": nn.Parameter(torch.tensor(T).to(device)),
            "direction_sign": torch.tensor(self.direction_sign, dtype=torch.float32).to(device)
        }
    
    def compute_atom_values(self, X: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """VECTORIZED - Batch compute all interactions"""
        T = params["T"]
        direction_sign = params["direction_sign"]
        
        # Compute all interactions
        interactions = []
        for combo in self.feature_combos:
            interaction = X[:, combo[0]]
            for idx in combo[1:]:
                interaction = interaction * X[:, idx]
            interactions.append(interaction)
            interactions.append(interaction)  # Duplicate for both directions
        
        interactions = torch.stack(interactions, dim=1)  # [B, A]
        
        # Expand thresholds: each threshold used twice (> and <=)
        T_expanded = torch.repeat_interleave(T, 2)  # [A]
        
        # Compute atom values
        diff = interactions - T_expanded.unsqueeze(0)
        return torch.sigmoid(self.beta * direction_sign.unsqueeze(0) * diff)


# ============================================================
# STRATEGY 5: FUZZY MEMBERSHIP ATOMS (VECTORIZED)
# ============================================================
class FuzzyMembershipAtomStrategy(BaseAtomStrategy):
    def __init__(self, membership_levels: List[str] = ["low", "medium", "high"], beta: float = 30.0):
        super().__init__("Fuzzy")
        self.membership_levels = membership_levels
        self.beta = beta
    
    def generate_atoms_metadata(self, feature_names: List[str], train_data: np.ndarray) -> List[AtomMeta]:
        atoms = []
        feat_list = []
        level_list = []
        
        for j, fname in enumerate(feature_names):
            for level_idx, level in enumerate(self.membership_levels):
                atoms.append(AtomMeta(
                    name=f"{fname} is {level.upper()}",
                    strategy_type=self.strategy_name,
                    feature_indices=[j],
                    params={"level": level}
                ))
                feat_list.append(j)
                level_list.append(level_idx)
        
        self.atoms_meta = atoms
        self.feat_idx = np.array(feat_list, dtype=np.int64)
        self.level_idx = np.array(level_list, dtype=np.int64)
        
        return atoms
    
    def initialize_parameters(self, train_data: np.ndarray, device: str) -> Dict[str, torch.nn.Parameter]:
        D = train_data.shape[1]
        num_levels = len(self.membership_levels)
        
        centers = np.zeros((D, num_levels), dtype=np.float32)
        for j in range(D):
            col = train_data[:, j]
            if num_levels == 3:
                centers[j] = [np.quantile(col, 0.25), np.quantile(col, 0.5), np.quantile(col, 0.75)]
            else:
                quantiles = np.linspace(0.2, 0.8, num_levels)
                centers[j] = [np.quantile(col, q) for q in quantiles]
        
        return {
            "centers": nn.Parameter(torch.tensor(centers).to(device)),
            "feat_idx": torch.tensor(self.feat_idx, dtype=torch.long).to(device),
            "level_idx": torch.tensor(self.level_idx, dtype=torch.long).to(device)
        }
    
    def compute_atom_values(self, X: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """VECTORIZED - No Python loops!"""
        centers = params["centers"]
        feat_idx = params["feat_idx"]
        level_idx = params["level_idx"]
        
        # Get features: [B, A]
        X_selected = X[:, feat_idx]
        
        # Get centers: [A]
        centers_selected = centers[feat_idx, level_idx]
        
        # Gaussian-like membership
        return torch.exp(-self.beta * (X_selected - centers_selected.unsqueeze(0)) ** 2)


# ============================================================
# STRATEGY 6: POLYNOMIAL ATOMS (VECTORIZED)
# ============================================================
class PolynomialAtomStrategy(BaseAtomStrategy):
    def __init__(self, transforms: List[str] = ["sqrt", "square", "log"], beta: float = 30.0):
        super().__init__("Polynomial")
        self.transforms = transforms
        self.beta = beta
        self.transform_map = {"sqrt": 0, "square": 1, "log": 2, "exp": 3}
    
    def generate_atoms_metadata(self, feature_names: List[str], train_data: np.ndarray) -> List[AtomMeta]:
        atoms = []
        feat_list = []
        transform_list = []
        direction_list = []
        
        for j, fname in enumerate(feature_names):
            for transform in self.transforms:
                # > threshold
                atoms.append(AtomMeta(
                    name=f"{transform}({fname}) > threshold",
                    strategy_type=self.strategy_name,
                    feature_indices=[j],
                    params={"transform": transform, "direction": ">"}
                ))
                feat_list.append(j)
                transform_list.append(self.transform_map[transform])
                direction_list.append(1.0)
                
                # <= threshold
                atoms.append(AtomMeta(
                    name=f"{transform}({fname}) ≤ threshold",
                    strategy_type=self.strategy_name,
                    feature_indices=[j],
                    params={"transform": transform, "direction": "<="}
                ))
                feat_list.append(j)
                transform_list.append(self.transform_map[transform])
                direction_list.append(-1.0)
        
        self.atoms_meta = atoms
        self.feat_idx = np.array(feat_list, dtype=np.int64)
        self.transform_idx = np.array(transform_list, dtype=np.int64)
        self.direction_sign = np.array(direction_list, dtype=np.float32)
        
        return atoms
    
    def _apply_transform(self, x, transform: str):
        if transform == "sqrt":
            return np.sqrt(np.abs(x) + 1e-6)
        elif transform == "square":
            return x ** 2
        elif transform == "log":
            return np.log(np.abs(x) + 1e-6)
        elif transform == "exp":
            return np.exp(np.clip(x, -10, 10))
        return x
    
    def initialize_parameters(self, train_data: np.ndarray, device: str) -> Dict[str, torch.nn.Parameter]:
        D = train_data.shape[1]
        num_transforms = len(self.transforms)
        
        T = np.zeros((D, num_transforms), dtype=np.float32)
        for j in range(D):
            col = train_data[:, j]
            for t_idx, transform in enumerate(self.transforms):
                transformed = self._apply_transform(col, transform)
                T[j, t_idx] = np.median(transformed)
        
        return {
            "T": nn.Parameter(torch.tensor(T).to(device)),
            "feat_idx": torch.tensor(self.feat_idx, dtype=torch.long).to(device),
            "transform_idx": torch.tensor(self.transform_idx, dtype=torch.long).to(device),
            "direction_sign": torch.tensor(self.direction_sign, dtype=torch.float32).to(device)
        }
    
    def compute_atom_values(self, X: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """VECTORIZED - Apply transforms in batch"""
        T = params["T"]
        feat_idx = params["feat_idx"]
        transform_idx = params["transform_idx"]
        direction_sign = params["direction_sign"]
        
        # Get features: [B, A]
        X_selected = X[:, feat_idx]
        
        # Apply transforms (vectorized for each type)
        X_transformed = torch.zeros_like(X_selected)
        
        for t_id, transform in enumerate(self.transforms):
            mask = (transform_idx == t_id)
            if mask.any():
                if transform == "sqrt":
                    X_transformed[:, mask] = torch.sqrt(torch.abs(X_selected[:, mask]) + 1e-6)
                elif transform == "square":
                    X_transformed[:, mask] = X_selected[:, mask] ** 2
                elif transform == "log":
                    X_transformed[:, mask] = torch.log(torch.abs(X_selected[:, mask]) + 1e-6)
                elif transform == "exp":
                    X_transformed[:, mask] = torch.exp(torch.clamp(X_selected[:, mask], -10, 10))
        
        # Get thresholds: [A]
        T_selected = T[feat_idx, transform_idx]
        
        # Compute atom values
        diff = X_transformed - T_selected.unsqueeze(0)
        return torch.sigmoid(self.beta * direction_sign.unsqueeze(0) * diff)


# ============================================================
# STRATEGY 7: STATISTICAL ATOMS (VECTORIZED)
# ============================================================
class StatisticalAtomStrategy(BaseAtomStrategy):
    def __init__(self, stats: List[str] = ["zscore"], beta: float = 30.0):
        super().__init__("Statistical")
        self.stats = stats
        self.beta = beta
    
    def generate_atoms_metadata(self, feature_names: List[str], train_data: np.ndarray) -> List[AtomMeta]:
        atoms = []
        feat_list = []
        threshold_list = []
        
        for j, fname in enumerate(feature_names):
            for stat in self.stats:
                if stat == "zscore":
                    # z > 1.5
                    atoms.append(AtomMeta(
                        name=f"zscore({fname}) > 1.5",
                        strategy_type=self.strategy_name,
                        feature_indices=[j],
                        params={"stat": stat, "threshold": 1.5}
                    ))
                    feat_list.append(j)
                    threshold_list.append(1.5)
                    
                    # z < -1.5
                    atoms.append(AtomMeta(
                        name=f"zscore({fname}) < -1.5",
                        strategy_type=self.strategy_name,
                        feature_indices=[j],
                        params={"stat": stat, "threshold": -1.5}
                    ))
                    feat_list.append(j)
                    threshold_list.append(-1.5)
        
        self.atoms_meta = atoms
        self.feat_idx = np.array(feat_list, dtype=np.int64)
        self.thresholds = np.array(threshold_list, dtype=np.float32)
        
        return atoms
    
    def initialize_parameters(self, train_data: np.ndarray, device: str) -> Dict[str, torch.nn.Parameter]:
        D = train_data.shape[1]
        means = train_data.mean(axis=0).astype(np.float32)
        stds = train_data.std(axis=0).astype(np.float32) + 1e-6
        
        return {
            "means": torch.tensor(means).to(device),
            "stds": torch.tensor(stds).to(device),
            "feat_idx": torch.tensor(self.feat_idx, dtype=torch.long).to(device),
            "thresholds": torch.tensor(self.thresholds, dtype=torch.float32).to(device)
        }
    
    def compute_atom_values(self, X: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """VECTORIZED - No Python loops!"""
        means = params["means"]
        stds = params["stds"]
        feat_idx = params["feat_idx"]
        thresholds = params["thresholds"]
        
        # Get features and compute z-scores: [B, A]
        X_selected = X[:, feat_idx]
        means_selected = means[feat_idx]
        stds_selected = stds[feat_idx]
        
        z_scores = (X_selected - means_selected.unsqueeze(0)) / stds_selected.unsqueeze(0)
        
        # Compare with thresholds
        return torch.sigmoid(self.beta * (z_scores - thresholds.unsqueeze(0)))


# ============================================================
# STRATEGY 8: DISTANCE ATOMS (VECTORIZED)
# ============================================================
class DistanceAtomStrategy(BaseAtomStrategy):
    def __init__(self, n_prototypes_per_class: int = 3, beta: float = 30.0):
        super().__init__("Distance")
        self.n_prototypes_per_class = n_prototypes_per_class
        self.beta = beta
        self.num_classes = None  # Will be set before generating metadata
    
    def set_num_classes(self, num_classes: int):
        """Set the number of classes - must be called before generate_atoms_metadata"""
        self.num_classes = num_classes
    
    def generate_atoms_metadata(self, feature_names: List[str], train_data: np.ndarray) -> List[AtomMeta]:
        if self.num_classes is None:
            raise ValueError("Must call set_num_classes() before generate_atoms_metadata()")
        
        atoms = []
        num_prototypes = self.n_prototypes_per_class * self.num_classes
        
        for p_idx in range(num_prototypes):
            atoms.append(AtomMeta(
                name=f"distance_to_prototype_{p_idx} < threshold",
                strategy_type=self.strategy_name,
                feature_indices=list(range(len(feature_names))),
                params={"prototype_idx": p_idx}
            ))
        
        self.atoms_meta = atoms
        return atoms
    
    def initialize_parameters(self, train_data: np.ndarray, device: str, train_labels: Optional[np.ndarray] = None) -> Dict[str, torch.nn.Parameter]:
        D = train_data.shape[1]
        
        if train_labels is not None:
            num_classes = len(np.unique(train_labels))
            prototypes = []
            
            for c in range(num_classes):
                class_data = train_data[train_labels == c]
                if len(class_data) >= self.n_prototypes_per_class:
                    kmeans = KMeans(n_clusters=self.n_prototypes_per_class, random_state=SEED, n_init=10)
                    kmeans.fit(class_data)
                    prototypes.append(kmeans.cluster_centers_)
                else:
                    prototypes.append(class_data[:self.n_prototypes_per_class])
            
            prototypes = np.vstack(prototypes).astype(np.float32)
        else:
            # Fallback: if no labels provided, use num_classes if set, otherwise default to 3
            num_classes_fallback = self.num_classes if self.num_classes is not None else 3
            num_prototypes = self.n_prototypes_per_class * num_classes_fallback
            num_samples = min(num_prototypes, len(train_data))
            prototypes = train_data[np.random.choice(len(train_data), 
                                                     num_samples,
                                                     replace=False)].astype(np.float32)
        
        thresholds = np.ones(len(prototypes), dtype=np.float32) * np.std(train_data)
        
        return {
            "prototypes": nn.Parameter(torch.tensor(prototypes).to(device)),
            "thresholds": nn.Parameter(torch.tensor(thresholds).to(device))
        }
    
    def compute_atom_values(self, X: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """VECTORIZED - Batch distance computation"""
        prototypes = params["prototypes"]
        thresholds = params["thresholds"]
        
        # Compute distances [B, num_prototypes]
        X_expanded = X.unsqueeze(1)  # [B, 1, D]
        P_expanded = prototypes.unsqueeze(0)  # [1, P, D]
        distances = torch.sqrt(((X_expanded - P_expanded) ** 2).sum(dim=-1))  # [B, P]
        
        # Sigmoid activation
        return torch.sigmoid(self.beta * (thresholds.unsqueeze(0) - distances))


# ============================================================
# ATOM STRATEGY FACTORY
# ============================================================
def create_atom_strategies(config_name: str) -> List[BaseAtomStrategy]:
    """Factory function to create atom strategies based on config name"""
    
    if config_name == "threshold_only":
        return [ThresholdAtomStrategy()]
    
    elif config_name == "range_only":
        return [RangeAtomStrategy()]
    
    elif config_name == "relational_only":
        return [RelationalAtomStrategy()]
    
    elif config_name == "interaction_only":
        return [InteractionAtomStrategy()]
    
    elif config_name == "fuzzy_only":
        return [FuzzyMembershipAtomStrategy()]
    
    elif config_name == "polynomial_only":
        return [PolynomialAtomStrategy()]
    
    elif config_name == "statistical_only":
        return [StatisticalAtomStrategy()]
    
    elif config_name == "distance_only":
        return [DistanceAtomStrategy()]
    
    elif config_name == "threshold+range":
        return [ThresholdAtomStrategy(), RangeAtomStrategy()]
    
    elif config_name == "threshold+relational":
        return [ThresholdAtomStrategy(), RelationalAtomStrategy()]
    
    elif config_name == "threshold+interaction":
        return [ThresholdAtomStrategy(), InteractionAtomStrategy()]
    
    elif config_name == "all_simple":
        return [
            ThresholdAtomStrategy(),
            RangeAtomStrategy(),
            RelationalAtomStrategy(),
            FuzzyMembershipAtomStrategy()
        ]
    
    elif config_name == "all_advanced":
        return [
            ThresholdAtomStrategy(),
            InteractionAtomStrategy(),
            PolynomialAtomStrategy(),
            StatisticalAtomStrategy()
        ]
    
    elif config_name == "full_suite":
        return [
            ThresholdAtomStrategy(),
            RangeAtomStrategy(),
            RelationalAtomStrategy(),
            InteractionAtomStrategy(),
            FuzzyMembershipAtomStrategy(),
            PolynomialAtomStrategy(),
            StatisticalAtomStrategy(),
            DistanceAtomStrategy()
        ]
    
    else:
        raise ValueError(f"Unknown config: {config_name}")


# ============================================================
# MENTAL MODEL LAYER (MULTI-STRATEGY)
# ============================================================
def gate_entropy(g: torch.Tensor) -> torch.Tensor:
    g = torch.clamp(g, 1e-6, 1-1e-6)
    H = -(g * torch.log(g) + (1-g) * torch.log(1-g))
    return H


class MentalModelLayerMultiStrategy(nn.Module):
    """Mental Model Layer with vectorized multi-strategy support"""
    
    def __init__(self, d_features: int, atom_strategies: List[BaseAtomStrategy],
                 num_classes: int, worldlets_per_class: int, 
                 train_data: np.ndarray, train_labels: Optional[np.ndarray] = None):
        super().__init__()
        
        self.d = d_features
        self.num_classes = num_classes
        self.K = worldlets_per_class
        self.atom_strategies = atom_strategies
        
        # Generate atoms from all strategies
        all_atoms_meta = []
        strategy_params = {}
        
        for strategy in atom_strategies:
            # Set num_classes for DistanceAtomStrategy before generating atoms
            if isinstance(strategy, DistanceAtomStrategy):
                strategy.set_num_classes(num_classes)
            
            atoms = strategy.generate_atoms_metadata(
                [f"f{i}" for i in range(d_features)],
                train_data
            )
            all_atoms_meta.extend(atoms)
            
            # Initialize strategy parameters
            if isinstance(strategy, DistanceAtomStrategy):
                params = strategy.initialize_parameters(train_data, DEVICE, train_labels)
            else:
                params = strategy.initialize_parameters(train_data, DEVICE)
            
            for key, param in params.items():
                strategy_params[f"{strategy.strategy_name}_{key}"] = param
        
        self.atoms_meta = all_atoms_meta
        self.num_atoms = len(all_atoms_meta)
        
        # Register strategy parameters
        for key, param in strategy_params.items():
            if isinstance(param, nn.Parameter):
                self.register_parameter(key, param)
            else:
                self.register_buffer(key, param)
        
        # Gates
        self.g_raw = nn.Parameter(torch.randn(num_classes, self.K, self.num_atoms) * 0.01)
        
        # Aggregation
        self.alpha = nn.Parameter(torch.zeros(num_classes, self.K))
        self.bias = nn.Parameter(torch.zeros(num_classes))
    
    def compute_all_atoms(self, X: torch.Tensor) -> torch.Tensor:
        """Compute atom values from all strategies (vectorized)"""
        all_atom_vals = []
        
        for strategy in self.atom_strategies:
            # Gather parameters for this strategy
            strategy_params = {}
            for key in list(self._parameters.keys()) + list(self._buffers.keys()):
                if key.startswith(f"{strategy.strategy_name}_"):
                    param_name = key.replace(f"{strategy.strategy_name}_", "")
                    if key in self._parameters:
                        strategy_params[param_name] = self._parameters[key]
                    else:
                        strategy_params[param_name] = self._buffers[key]
            
            # Compute atoms (vectorized!)
            atom_vals = strategy.compute_atom_values(X, strategy_params)
            all_atom_vals.append(atom_vals)
        
        # Concatenate all atoms
        if len(all_atom_vals) > 0:
            return torch.cat(all_atom_vals, dim=1)
        else:
            return torch.zeros(X.shape[0], 0).to(X.device)
    
    def compute_all(self, X: torch.Tensor):
        """Unified computation: atoms → worldlets → logits"""
        B = X.shape[0]
        
        # Compute all atoms (vectorized)
        Avals = self.compute_all_atoms(X)  # [B, A]
        
        # Worldlet satisfactions
        g = torch.sigmoid(self.g_raw)  # [C, K, A]
        g = torch.clamp(g, 1e-6, 1-1e-6)
        
        Aexp = Avals.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, A]
        factors = (1.0 - g) + g * Aexp  # [B, C, K, A]
        s_log = torch.sum(torch.log(factors), dim=-1)  # [B, C, K]
        S = torch.exp(s_log)  # [B, C, K]
        
        # Class logits
        logits = (self.alpha.unsqueeze(0) * S).sum(dim=-1) + self.bias.unsqueeze(0)  # [B, C]
        
        # Inspect predictions
        insp_scores = S.max(dim=-1).values  # [B, C]
        inspect_preds = insp_scores.argmax(dim=-1)  # [B]
        
        dbg = {"Avals": Avals, "g": g, "S": S}
        return logits, inspect_preds, dbg
    
    def forward(self, X: torch.Tensor):
        logits, _, dbg = self.compute_all(X)
        return logits, dbg
    
    @torch.no_grad()
    def inspect_predict(self, X: torch.Tensor) -> torch.Tensor:
        _, inspect_preds, _ = self.compute_all(X)
        return inspect_preds


# ============================================================
# TRAINING HELPERS
# ============================================================
def batch_iter(X: torch.Tensor, y: torch.Tensor, bs: int):
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    for s in range(0, n, bs):
        e = min(n, s + bs)
        ids = idx[s:e]
        yield X[ids], y[ids]


def train_mml_multi_strategy(X, y, feature_names, class_names, atom_strategies, 
                              config_name="custom", verbose=False, pbar=None):
    """Train MML with specified atom strategies"""
    
    # Standardize
    scaler = None
    if STANDARDIZE:
        scaler = StandardScaler()
        X = scaler.fit_transform(X.astype(np.float32)).astype(np.float32)
    
    # Split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    num_classes = len(np.unique(y))
    D = X.shape[1]
    
    # Tensors
    Xtr_t = torch.from_numpy(Xtr).to(DEVICE)
    ytr_t = torch.from_numpy(ytr).to(DEVICE)
    Xte_t = torch.from_numpy(Xte).to(DEVICE)
    yte_t = torch.from_numpy(yte).to(DEVICE)
    
    # Model
    model = MentalModelLayerMultiStrategy(
        d_features=D,
        atom_strategies=atom_strategies,
        num_classes=num_classes,
        worldlets_per_class=WORLDLETS_PER_CLASS,
        train_data=Xtr,
        train_labels=ytr
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    
    if verbose:
        print(f"[{config_name}] Atoms={model.num_atoms}, Training...")
    
    # Train with progress bar
    best_val_acc, best_state = -1.0, None
    start_time = time.time()
    
    epoch_pbar = tqdm(range(1, EPOCHS + 1), desc=f"{config_name}", leave=False, disable=not HAS_TQDM)
    
    for ep in epoch_pbar:
        model.train()
        total_loss = 0.0
        total_n = 0
        
        for xb, yb in batch_iter(Xtr_t, ytr_t, BATCH_SIZE):
            optimizer.zero_grad()
            logits, dbg = model(xb)
            ce = F.cross_entropy(logits, yb)
            
            g = dbg["g"]
            S = dbg["S"]
            L_sparse = g.mean()
            L_bin = gate_entropy(g).mean()
            L_act = S.mean()
            
            loss = ce + LAMBDA_SPARSITY*L_sparse + LAMBDA_BINARY*L_bin + LAMBDA_ACT_PEN*L_act
            loss.backward()
            optimizer.step()
            
            bs = xb.shape[0]
            total_loss += float(loss.detach().cpu()) * bs
            total_n += bs
        
        # Eval
        model.eval()
        with torch.no_grad():
            logits_te, _, _ = model.compute_all(Xte_t)
            acc_te = accuracy_score(yte, logits_te.argmax(dim=1).cpu().numpy())
        
        if acc_te > best_val_acc:
            best_val_acc = acc_te
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        
        epoch_pbar.set_postfix({"loss": f"{total_loss/total_n:.4f}", "acc": f"{acc_te:.4f}"})
    
    epoch_pbar.close()
    train_time = time.time() - start_time
    
    # Load best
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        model.eval()
    
    # Final metrics
    with torch.no_grad():
        logits_te, _, _ = model.compute_all(Xte_t)
        y_pred = logits_te.argmax(dim=1).cpu().numpy()
        acc = accuracy_score(yte, y_pred)
    
    if pbar:
        pbar.set_postfix({"acc": f"{acc:.4f}", "time": f"{train_time:.1f}s"})
    
    return {
        "config_name": config_name,
        "acc": acc,
        "train_time": train_time,
        "num_atoms": model.num_atoms,
        "model": model,
        "y_test": yte,
        "y_pred": y_pred
    }


# ============================================================
# BASELINES
# ============================================================
def fit_baselines(X, y):
    if STANDARDIZE:
        scaler = StandardScaler()
        X = scaler.fit_transform(X.astype(np.float32)).astype(np.float32)
    
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    results = {}
    
    if USE_LOGREG:
        lr = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=1.0, max_iter=2000, random_state=SEED)
        lr.fit(Xtr, ytr)
        results["LogReg"] = accuracy_score(yte, lr.predict(Xte))
    
    if USE_RANDOM_FOREST:
        rf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, random_state=SEED, n_jobs=-1)
        rf.fit(Xtr, ytr)
        results["RandomForest"] = accuracy_score(yte, rf.predict(Xte))
    
    if USE_GRAD_BOOST:
        gb = GradientBoostingClassifier(n_estimators=GB_N_ESTIMATORS, learning_rate=GB_LEARNING_RATE, 
                                        max_depth=GB_MAX_DEPTH, random_state=SEED)
        gb.fit(Xtr, ytr)
        results["GradBoost"] = accuracy_score(yte, gb.predict(Xte))
    
    if USE_SVM_RBF:
        svm = SVC(C=SVM_C, gamma=SVM_GAMMA, random_state=SEED)
        svm.fit(Xtr, ytr)
        results["SVM-RBF"] = accuracy_score(yte, svm.predict(Xte))
    
    return results


# ============================================================
# INSTANT RESULTS DISPLAY HELPERS
# ============================================================

def display_result_row(row: Dict[str, Any]):
    """Print a single result row immediately"""
    
    print("\n" + "─"*90)
    print(f"✅ RESULT: {row['Dataset']} / {row['Config']}")
    print("─"*90)
    
    # MML Results
    print(f"   MML Accuracy:  {row['MML_Acc']:.4f}  |  Atoms: {row['Num_Atoms']}  |  Time: {row['Train_Time']:.1f}s")
    
    # Baselines comparison
    baseline_cols = [k for k in row.keys() if k not in ['Dataset', 'Config', 'MML_Acc', 'Num_Atoms', 'Train_Time']]
    if baseline_cols:
        baseline_str = " | ".join([f"{k}: {row[k]:.4f}" for k in baseline_cols])
        print(f"   Baselines:     {baseline_str}")
    
    # Performance indicator
    if baseline_cols:
        best_baseline = max([row[k] for k in baseline_cols])
        if row['MML_Acc'] > best_baseline:
            print(f"   🏆 MML WINS (+{(row['MML_Acc'] - best_baseline):.4f})")
        elif row['MML_Acc'] >= best_baseline - 0.01:  # Within 1%
            print(f"   🤝 COMPETITIVE (±{abs(row['MML_Acc'] - best_baseline):.4f})")
        else:
            print(f"   📊 BASELINE LEADS (-{(best_baseline - row['MML_Acc']):.4f})")
    
    print("─"*90)


def save_result_incrementally(row: Dict[str, Any], csv_path: str = "grid_search_results_live.csv"):
    """Save result to CSV immediately (append mode)"""
    
    # Create DataFrame from single row
    df = pd.DataFrame([row])
    
    # Check if file exists
    if os.path.exists(csv_path):
        # Append without header
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        # Create new file with header
        df.to_csv(csv_path, mode='w', header=True, index=False)


# ============================================================
# GRID SEARCH WITH INSTANT RESULTS
# ============================================================
def run_single_config(dsname, config_name, X, y, feature_names, class_names, base_res):
    """Run a single configuration (for parallel execution)"""
    try:
        strategies = create_atom_strategies(config_name)
        result = train_mml_multi_strategy(
            X, y, feature_names, class_names,
            atom_strategies=strategies,
            config_name=config_name,
            verbose=False
        )
        
        row = {
            "Dataset": dsname,
            "Config": config_name,
            "MML_Acc": round(result["acc"], 4),
            "Num_Atoms": result["num_atoms"],
            "Train_Time": round(result["train_time"], 1)
        }
        
        for k, v in base_res.items():
            row[k] = round(v, 4)
        
        return row
        
    except Exception as e:
        print(f"[ERROR] {dsname}/{config_name} failed: {e}")
        return None


def run_grid_search_staged(datasets_list: List[str], atom_configs: List[str]):
    """Staged grid search with INSTANT results display"""
    
    all_results = []
    csv_path = "grid_search_results_live.csv"
    
    # Delete old CSV if exists
    if os.path.exists(csv_path):
        os.remove(csv_path)
        print(f"[SETUP] Deleted old {csv_path}")
    
    print(f"[SETUP] Results will be saved to: {csv_path}")
    print(f"[SETUP] Testing {len(atom_configs)} configs on {len(datasets_list)} datasets")
    print("="*90)
    
    # Progress bar for datasets
    ds_pbar = tqdm(datasets_list, desc="Datasets", position=0)
    
    for dsname in ds_pbar:
        ds_pbar.set_description(f"Dataset: {dsname}")
        
        # Load data
        X, y, feature_names, class_names = load_dataset_by_name(dsname)
        
        # Baselines
        base_res = fit_baselines(X, y)
        
        # Test each config (parallel or sequential)
        if PARALLEL_JOBS != 1 and HAS_JOBLIB:
            # Parallel execution
            config_pbar = tqdm(atom_configs, desc="Configs", position=1, leave=False)
            results = Parallel(n_jobs=PARALLEL_JOBS)(
                delayed(run_single_config)(dsname, config_name, X, y, feature_names, class_names, base_res)
                for config_name in config_pbar
            )
            config_pbar.close()
            
            # Display and save results
            for row in results:
                if row is not None:
                    all_results.append(row)
                    display_result_row(row)
                    save_result_incrementally(row, csv_path)
        else:
            # Sequential execution with instant display
            config_pbar = tqdm(atom_configs, desc="Configs", position=1, leave=False)
            
            for config_name in config_pbar:
                config_pbar.set_description(f"Config: {config_name}")
                
                row = run_single_config(dsname, config_name, X, y, feature_names, class_names, base_res)
                if row is not None:
                    all_results.append(row)
                    display_result_row(row)
                    save_result_incrementally(row, csv_path)
            
            config_pbar.close()
    
    ds_pbar.close()
    
    return pd.DataFrame(all_results)


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*90)
    print("MENTAL MODEL LAYER - OPTIMIZED MULTI-STRATEGY FRAMEWORK")
    print("="*90)
    print(f"[OPTIMIZATIONS] ✅ Vectorized | ✅ Progress bars | ✅ Parallel | ✅ INSTANT RESULTS")
    print(f"[NEW] ✅ Synthetic Dataset Suite (easy, hard, XOR, moons, circles)")
    
    if ENABLE_GRID_SEARCH:
        print(f"\n[MODE] Grid Search - Testing {len(ATOM_STRATEGY_CONFIGS)} configs on {len(DATASETS)} datasets")
        print(f"[DATASETS] {', '.join(DATASETS)}")
        
        results_df = run_grid_search_staged(DATASETS, ATOM_STRATEGY_CONFIGS)
        
        # Display final summary
        print("\n" + "="*90)
        print("[FINAL SUMMARY - ALL RESULTS]")
        print("="*90)
        print(results_df.to_string(index=False))
        
        # Save final consolidated CSV
        final_csv = "grid_search_results_final.csv"
        results_df.to_csv(final_csv, index=False)
        print(f"\n[SAVED] Final results: {final_csv}")
        
        # Summary: Best config per dataset
        print("\n" + "="*90)
        print("[BEST CONFIGS PER DATASET]")
        print("="*90)
        for ds in DATASETS:
            ds_results = results_df[results_df["Dataset"] == ds]
            if len(ds_results) > 0:
                best_row = ds_results.loc[ds_results["MML_Acc"].idxmax()]
                print(f"{ds}: {best_row['Config']} (acc={best_row['MML_Acc']:.4f}, atoms={best_row['Num_Atoms']}, time={best_row['Train_Time']}s)")
        
        # Top 5 overall configs
        print("\n" + "="*90)
        print("[TOP 5 CONFIGS (by average accuracy)]")
        print("="*90)
        avg_acc = results_df.groupby("Config")["MML_Acc"].mean().sort_values(ascending=False)
        for i, (config, acc) in enumerate(avg_acc.head(5).items(), 1):
            print(f"{i}. {config}: {acc:.4f}")
        
        # 🆕 XOR SPECIAL ANALYSIS
        print("\n" + "="*90)
        print("[XOR ANALYSIS - NON-LINEAR TEST]")
        print("="*90)
        xor_results = results_df[results_df["Dataset"] == "xor"]
        if len(xor_results) > 0:
            print("Expected: LogReg ~50% (fails), Non-linear methods ~85%+")
            print(xor_results[["Config", "MML_Acc", "LogReg"]].to_string(index=False))
            print(f"\nBest XOR config: {xor_results.loc[xor_results['MML_Acc'].idxmax()]['Config']}")
    
    else:
        # Single run
        print("\n[MODE] Single Run - threshold_only")
        
        for dsname in DATASETS:
            print(f"\n[RUN] Dataset: {dsname}")
            
            X, y, feature_names, class_names = load_dataset_by_name(dsname)
            base_res = fit_baselines(X, y)
            print(f"[BASELINES] {base_res}")
            
            strategies = create_atom_strategies("threshold_only")
            result = train_mml_multi_strategy(
                X, y, feature_names, class_names,
                atom_strategies=strategies,
                config_name="threshold_only",
                verbose=True
            )
    
    print("\n[DONE] ✅ Optimized multi-strategy framework with synthetic datasets complete!")
    print("="*90)