import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.utils.data as data_utils
import torch.nn.functional as F
import mlflow
import mlflow.pytorch
import mlflow.data as mlfdata
import matplotlib.pyplot as plt
import yaml
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from data.data_management.dataset_manager import DatasetReader
from eval.scripts.metrics import calculate_metrics, calculate_counting_metrics, calculate_score_quantiles
from models.model import Attention, AttentionBatchNorm, AttentionDropout, AttentionThirdConv, GatedAttention
from models.fpn_mil_model import FPNMIL
from models.clam_model import CLAM
from models.learned_grayscale import LearnedGrayscale
import visualize_features as vf


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

# Digit treated as the positive class when MNIST bags carry multi-class labels
POSITIVE_MNIST_DIGIT = 9

# Color cycle for the aggregated per-seed counting scatter plot
SEED_PLOT_COLORS = ['red', 'blue', 'green', 'orange', 'purple',
                    'cyan', 'magenta', 'yellow', 'black', 'brown']

# Threshold sweep grid for the CLAM counting evaluation
COUNT_SWEEP_THRESHOLDS = np.arange(0.05, 0.75, 0.05)

# Reference RGB->gray conversions the learned grayscale weights are compared to
GRAYSCALE_REFERENCES = {
    "bt601": torch.tensor([0.299, 0.587, 0.114]),
    "bt709": torch.tensor([0.2126, 0.7152, 0.0722]),
    "R":     torch.tensor([1., 0., 0.]),
    "G":     torch.tensor([0., 1., 0.]),
    "B":     torch.tensor([0., 0., 1.]),
    "mean":  torch.tensor([1 / 3, 1 / 3, 1 / 3]),
}


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def binarize_instance_labels(labels):
    """Map instance labels to a binary positive/negative encoding.

    Multi-class labels (MNIST digits) are reduced to ``label == POSITIVE_MNIST_DIGIT``;
    labels that are already binary are passed through unchanged.

    Args:
        labels: array-like of instance labels (any shape, gets flattened).
    Returns:
        1-D int numpy array of 0/1 labels.
    """
    arr = np.asarray(labels).flatten()
    if arr.size and arr.max() > 1:
        return (arr == POSITIVE_MNIST_DIGIT).astype(int)
    return arr.astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# CLI and configuration
# ══════════════════════════════════════════════════════════════════════════════

def build_arg_parser():
    """Create the full command line parser for a MIL training run."""
    parser = argparse.ArgumentParser(description='Testing Atteintion MIL models on datasets loaded from H5 files.')

    # ── Base parameters ───────────────────────────────────────────────────────
    parser.add_argument('--config', type=str, default=None, metavar='CONFIG',
                        help='path to YAML config file (default: None)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                        help='weight decay')
    parser.add_argument('--naive_counting', action='store_true', default=False,
                        help='activates counting of positve instances for model testing')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1], metavar='S',
                        help='list of random seeds (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    # ── Model parameters ──────────────────────────────────────────────────────
    parser.add_argument('--model', type=str, default='attention',
                        help='Choose b/w attention and gated_attention')
    parser.add_argument('--model_M', type=int, default=500,
                        help='Dimensionality of the MLP layer in the attention mechanism (default: 500)')
    parser.add_argument('--model_L', type=int, default=128,
                        help='Dimensionality of the output of the attention mechanism (default: 128)')
    parser.add_argument('--model_pool_size', type=int, default=4,
                        help='Output size of the adaptive pooling layer (default: 4)')
    parser.add_argument('--model_num_maps', type=int, default=50,
                        help='Number of feature maps output by the convolutional layers (default: 50)')
    parser.add_argument('--model_kernel_size', type=int, default=5,
                        help='Kernel size for convolutional layers (default: 5)')
    parser.add_argument('--model_num_scales', type=int, default=3,
                        help='FPN-MIL: number of pyramid scales / backbone stages (default: 3)')
    parser.add_argument('--attention_activation', type=str, default='softmax',
                        choices=['sigmoid', 'min_max', 'softmax', 'sparsemax', "softmax_temperature", "entmax"],
                        help='activation function for attention weights (default: softmax)')
    parser.add_argument('--rgb', action='store_true', default=False,
                        help='use RGB input instead of grayscale')
    parser.add_argument('--grayscaling', action='store_true', default=False,
                        help='use learned grayscale conversion for RGB input (rgb and grayscale are exclusive)')
    parser.add_argument('--pretrained_backbone', type=str, default=None,
                        help='path to pretrained backbone weights')
    parser.add_argument('--save_model', type=str, default=None, metavar='PATH',
                        help='save the trained state_dict after testing; PATH is a directory '
                             '(filename is generated) or a .pth file. Der Seed wird immer an '
                             'den Dateinamen angehaengt (default: None)')
    parser.add_argument('--load_model', type=str, default=None, metavar='PATH',
                        help='load a state_dict into the model before training (default: None)')
    parser.add_argument('--eval_only', action='store_true', default=False,
                        help='skip training and only run the test loop (use with --load_model)')

    # ── FPN parameters ────────────────────────────────────────────────────────
    parser.add_argument('--model_dx', type=int, default=256,
                        help='FPN-MIL: shared FPN channel dimension d_x (default: 256)')
    parser.add_argument('--model_base_channels', type=int, default=32,
                        help='FPN-MIL: base channel count of the backbone (default: 32)')

    # ── CLAM parameters ───────────────────────────────────────────────────────
    parser.add_argument('--clam_k_sample', type=int, default=8,
                        help='Anzahl Top-/Bottom-Instanzen fuer CLAM Instance Clustering')
    parser.add_argument('--clam_bag_weight', type=float, default=0.7,
                        help='Gewicht des Bag-Loss im kombinierten CLAM-Loss')
    parser.add_argument('--clam_pseudo_threshold', action='store_true', default=False,
                        help='Use pseudo-thresholding for instance loss in CLAM (default: False)')
    parser.add_argument('--clam_pseudo_quantile_pos', type=float, default=0.5,
                        help='CLAM pseudo-threshold: Quantil, ab dem Instanzen pseudo-positiv gelabelt werden (default: 0.5)')
    parser.add_argument('--clam_pseudo_quantile_neg', type=float, default=0.25,
                        help='CLAM pseudo-threshold: Quantil, bis zu dem Instanzen pseudo-negativ gelabelt werden (default: 0.25)')
    parser.add_argument('--count_threshold_eval', action='store_true', default=False,
                        help='CLAM: vergleicht Count-Threshold-Strategien (Sweep, Otsu, Val-kalibriert, Baseline) im Test; alle bag-gegatet')
    parser.add_argument('--calibrate_count_threshold', action='store_true', default=False,
                        help='CLAM: kalibriert den Zaehl-Threshold auf dem Val-Split und benutzt ihn '
                             'fuer count_positive_instances im Test (statt Default 0.5)')
    parser.add_argument('--soft_counting', action='store_true', default=False,
                        help='CLAM: zusaetzlicher Soft-Count (Summe der Instanz-Wahrscheinlichkeiten, bag-gegatet) in count_threshold_eval')

    # ── Data parameters ───────────────────────────────────────────────────────
    parser.add_argument('--dataset', type=str, default='mnist_bags', metavar='H5',
                        help='path to H5 file containing the dataset (default: mnist_bags.h5)')
    parser.add_argument('--path', type=str, default='../data/datasets/bags/mnist_bags.h5', metavar='H5',
                        help='path to H5 file containing the dataset (default: mnist_bags.h5)')

    # ── MLflow parameters ─────────────────────────────────────────────────────
    parser.add_argument('--exp_name', type=str, default=None, metavar='EXP',
                        help='name of the MLflow experiment (default: default)')
    parser.add_argument('--run_name', type=str, default=None, metavar='RUN',
                        help='name of the MLflow run (default: None)')
    parser.add_argument('--log_attention_weights', action='store_true', default=False,
                        help='log attention weights as artifact in MLflow')
    parser.add_argument('--log_instance_scores', action='store_true', default=False,
                        help='CLAM: log per-bag instance classifier scores and instance labels as artifact in MLflow')
    parser.add_argument('--visualize_features', action='store_true', default=False,
                        help='visualize extracted features using UMAP and log the plot to MLflow')

    return parser


def parse_args():
    """Parse CLI arguments, applying a YAML config file as defaults if given.

    Returns:
        (args, config_path): parsed arguments (with ``args.cuda`` resolved) and
        the path of the YAML config that was used, or None.
    """
    # ── Pre-pass: find --config before the main parser builds its defaults ────
    config_parser = argparse.ArgumentParser(description='Config file parser', add_help=False)
    config_parser.add_argument('--config', type=str, default=None, metavar='CONFIG')
    config_args, _ = config_parser.parse_known_args()

    parser = build_arg_parser()

    # ── Read configuration from YAML file if provided ─────────────────────────
    if config_args.config is not None:
        if os.path.exists(config_args.config):
            print(f"Loading configuration from {config_args.config}")
            with open(config_args.config, 'r') as f:
                yaml_config = yaml.safe_load(f)
            # Update default arguments with values from YAML config
            parser.set_defaults(**yaml_config)

    # ── Parse the final arguments ─────────────────────────────────────────────
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args, config_args.config


def save_run_config(args):
    """Dump the effective run configuration to configs/ so it can be replayed.

    Returns:
        Path of the written YAML file.
    """
    print(f"No config file provided. Using default arguments and command line overrides.")
    args_dict = vars(args)
    config_filename = (f"configs/{args.model}_{args.dataset}_lr{args.lr}_reg{args.reg}"
                       f"_ep{args.epochs}_attention_activation{args.attention_activation}_config.yaml")
    with open(config_filename, 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False)
    print(f"Run configuration saved to {config_filename}")
    return config_filename


# ══════════════════════════════════════════════════════════════════════════════
# Run context
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RunContext:
    """Everything a single seed run needs: config, model, data and shared buffers."""

    # ── Configuration and seed ────────────────────────────────────────────────
    args: argparse.Namespace
    seed: int

    # ── Model and optimization ────────────────────────────────────────────────
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer

    # ── Data loaders ──────────────────────────────────────────────────────────
    train_loader: Any
    val_loader: Any
    test_loader: Any

    # ── Derived logging flags ─────────────────────────────────────────────────
    log_instance_scores: bool = False

    # ── Counting-threshold evaluation state (CLAM only) ───────────────────────
    p_train: Optional[float] = None            # positive instance fraction of the train set (baseline)
    val_scores_per_bag: list = field(default_factory=list)   # instance scores per val bag (last epoch)
    val_true_counts: list = field(default_factory=list)      # true counts per val bag
    val_pred_pos_per_bag: list = field(default_factory=list)  # bag prediction per val bag (calibration gate)
    count_threshold: Optional[float] = None    # auf Val kalibrierter Zaehl-Threshold (None = nicht kalibriert)

    @property
    def is_clam(self):
        return self.args.model == 'clam'

    @property
    def device(self):
        return "cuda" if self.args.cuda else "cpu"


@dataclass
class TestBuffers:
    """Per-bag values collected during the test loop and consumed by the metric helpers."""

    # ── Bag-level accumulators ────────────────────────────────────────────────
    test_loss: float = 0.
    test_error: float = 0.
    y_true: list = field(default_factory=list)
    y_pred: list = field(default_factory=list)
    y_prob: list = field(default_factory=list)

    # ── Naive counting ────────────────────────────────────────────────────────
    count_truth: list = field(default_factory=list)
    count_pred: list = field(default_factory=list)

    # ── Artifact aggregation (attention weights / instance scores) ────────────
    attention_agg: list = field(default_factory=list)
    inst_score_agg: list = field(default_factory=list)   # instance scores per test bag
    inst_label_agg: list = field(default_factory=list)   # matching instance labels per test bag

    # ── Patch-level metrics: attention-based models ───────────────────────────
    all_instance_labels: list = field(default_factory=list)
    all_attention_weights: list = field(default_factory=list)
    all_thresholds: list = field(default_factory=list)

    # ── Patch-level metrics: CLAM instance classifier ─────────────────────────
    patch_inst_labels: list = field(default_factory=list)
    patch_inst_scores: list = field(default_factory=list)

    # ── Counting-threshold evaluation (CLAM only) ─────────────────────────────
    scores_per_bag: list = field(default_factory=list)   # instance scores per test bag
    true_counts: list = field(default_factory=list)      # true counts per test bag
    pred_pos_per_bag: list = field(default_factory=list)  # bag prediction per test bag (gate)


# ══════════════════════════════════════════════════════════════════════════════
# Setup: data, model, optimizer, result containers
# ══════════════════════════════════════════════════════════════════════════════

def build_dataloaders(args):
    """Create the train/validation/test loaders for the configured H5 dataset.

    Returns:
        (train_loader, val_loader, test_loader), all with batch size 1 (one bag per step).
    """
    print('Load Train and Test Set')
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    train_loader = data_utils.DataLoader(DatasetReader(args.path, dataset_name=args.dataset, split='train'),
                                         batch_size=1,
                                         shuffle=True,
                                         **loader_kwargs)
    val_loader = data_utils.DataLoader(DatasetReader(args.path, dataset_name=args.dataset, split='validation'),
                                       batch_size=1,
                                       shuffle=False,
                                       **loader_kwargs)
    test_loader = data_utils.DataLoader(DatasetReader(args.path, dataset_name=args.dataset, split='test'),
                                        batch_size=1,
                                        shuffle=False,
                                        **loader_kwargs)
    return train_loader, val_loader, test_loader


def _common_model_tags(model):
    """MLflow tags shared by all conv-backbone attention models."""
    return {
        "model_M": model.M,
        "model_L": model.L,
        "model_pool_size": model.pool_size,
        "model_num_maps": model.num_maps,
        "model_kernel_size": model.kernel_size,
    }


def build_model(args):
    """Instantiate the model selected via --model and collect its MLflow tags.

    Returns:
        (model, model_tags)
    """
    print('Initialize model')

    # ── Constructor kwargs shared by all conv-backbone attention models ───────
    common_kwargs = dict(M=args.model_M, L=args.model_L,
                         num_maps=args.model_num_maps,
                         kernel_size=args.model_kernel_size,
                         pool_size=args.model_pool_size,
                         in_channels=3 if args.rgb else 1)

    # ── Model-specific construction ───────────────────────────────────────────
    if args.model == 'attention':
        model = Attention(**common_kwargs, grayscaling=args.grayscaling,
                          attention_activation=args.attention_activation)
        model_tags = {**_common_model_tags(model),
                      "model_attention_branches": model.ATTENTION_BRANCHES,
                      "model_architecture": "Conv2d(1->20->50) -> FC(800->M->L)"}

    elif args.model == 'gated_attention':
        model = GatedAttention(**common_kwargs, attention_activation=args.attention_activation)
        model_tags = {**_common_model_tags(model),
                      "model_attention_branches": model.ATTENTION_BRANCHES,
                      "model_architecture": "GatedAttention: V(Tanh) + U(Sigmoid) + w"}

    elif args.model == 'attention_batchnorm':
        model = AttentionBatchNorm(**common_kwargs, attention_activation=args.attention_activation)
        model_tags = {**_common_model_tags(model),
                      "model_attention_branches": model.ATTENTION_BRANCHES,
                      "model_architecture": "AttentionBatchNorm: Conv2d(1->20->50) + BatchNorm -> FC(800->M->L)"}

    elif args.model == 'attention_third_conv':
        model = AttentionThirdConv(**common_kwargs, attention_activation=args.attention_activation)
        model_tags = {**_common_model_tags(model),
                      "model_attention_branches": model.ATTENTION_BRANCHES,
                      "model_architecture": "AttentionThirdConv: Conv2d(1->20->50) -> FC(800->M->L)"}

    elif args.model == 'attention_dropout':
        model = AttentionDropout(**common_kwargs, attention_activation=args.attention_activation)
        model_tags = {**_common_model_tags(model),
                      "model_attention_branches": model.ATTENTION_BRANCHES,
                      "model_architecture": "AttentionDropout: Conv2d(1->20->50) -> FC(800->M->L) + Dropout(p=0.25)"}

    elif args.model == 'fpn_mil':
        model = FPNMIL(in_channels=3 if args.rgb else 1,
                       base_channels=args.model_base_channels,
                       num_scales=args.model_num_scales,
                       d_x=args.model_dx, d=args.model_M, L=args.model_L,
                       kernel_size=args.model_kernel_size, gated=True)
        # FPN-MIL exposes a different attribute set, so its tags are built from args
        model_tags = {
            "model_M": args.model_M,
            "model_L": args.model_L,
            "model_kernel_size": args.model_kernel_size,
            "model_num_scales": args.model_num_scales,
            "model_dx": args.model_dx,
            "model_base_channels": args.model_base_channels,
            "model_architecture": (f"FPN-MIL: {args.model_num_scales} scales, d_x={args.model_dx}, "
                                   f"gated AbMIL + multi-scale aggregator"),
        }

    elif args.model == 'clam':
        model = CLAM(**common_kwargs,
                     k_sample=args.clam_k_sample,
                     pseudo_threshold=args.clam_pseudo_threshold,
                     dropout=0.25,
                     grayscaling=args.grayscaling,
                     pseudo_quantile_pos=args.clam_pseudo_quantile_pos,
                     pseudo_quantile_neg=args.clam_pseudo_quantile_neg)
        model_tags = {**_common_model_tags(model),
                      "model_k_sample": model.k_sample,
                      "model_pseudo_threshold": model.pseudo_threshold,
                      "model_pseudo_quantile_pos": model.pseudo_quantile_pos,
                      "model_pseudo_quantile_neg": model.pseudo_quantile_neg,
                      "model_architecture": "CLAM_SB (gated attn + instance classifier)"}

    else:
        raise ValueError(f"Unknown model type: {args.model}")

    return model, model_tags


def load_pretrained_backbone(model, weights_path):
    """Load pretrained backbone weights into the model's feature extractor."""
    print(f"Loading pretrained backbone weights from {weights_path}")
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
    if 'backbone' in state_dict:
        model.backbone.load_state_dict(state_dict['backbone'], strict=True)
    else:
        model.backbone.load_state_dict(state_dict, strict=True)
    print("Pretrained backbone weights loaded successfully.")


def save_model_state(model, args, seed):
    """Save the model's state_dict to disk (and log it as MLflow artifact).

    ``args.save_model`` may be a directory (the filename is then generated from
    model/dataset) or a ``.pth``/``.pt`` file path. The seed is always appended
    so runs over several seeds do not overwrite each other.

    Returns:
        Path of the written checkpoint file.
    """
    target = args.save_model
    if target.endswith(('.pth', '.pt')):
        stem, ext = os.path.splitext(target)
    else:
        stem = os.path.join(target, f"{args.model}_{args.dataset}")
        ext = '.pth'
    path = f"{stem}_seed{seed}{ext}"

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model state_dict gespeichert: {path}")

    mlflow.log_artifact(path, artifact_path="state_dicts")
    return path


def load_model_state(model, weights_path, device):
    """Load a state_dict saved by ``save_model_state`` into ``model``."""
    print(f"Lade state_dict aus {weights_path}")
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    print("State_dict erfolgreich geladen.")


def build_optimizer(model, args):
    """Create the Adam optimizer.

    With a pretrained backbone the backbone gets a 100x smaller learning rate
    than the freshly initialized head (and no weight decay).
    """
    if args.pretrained_backbone:
        backbone_params = list(model.backbone.parameters())
        head_params = [p for n, p in model.named_parameters()
                       if not n.startswith("backbone.")]
        return torch.optim.Adam([
            {'params': backbone_params, 'lr': args.lr * 0.01},
            {'params': head_params, 'lr': args.lr},
        ])

    return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)


def init_results_container(args, log_instance_scores):
    """Create the cross-seed result dict that is logged as an MLflow table."""
    results = {
        "seeds": [],
        "bag_ids": [],
        "truth": [],
        "predicted": [],
    }
    if args.naive_counting:
        results["count_truth"] = []
        results["count_pred"] = []
        results["count_threshold"] = []

    if args.log_attention_weights:
        results["attention_weights"] = []

    # Instance scores of the CLAM instance classifier per test bag (like attention_weights)
    if log_instance_scores:
        results["instance_scores"] = []
        results["instance_labels"] = []

    return results


def compute_train_positive_fraction(train_loader):
    """Mean fraction of positive instances per bag over the training set.

    Used as the trivial baseline estimator (p_train * N) in the counting evaluation.

    Returns:
        float, or None if no bag contained any instance.
    """
    print('Bestimme Baseline-Positiv-Anteil p_train ueber das Trainingsset...')
    pos_fracs = []
    for _p, _c, _lbl, _cnt, inst_lbl in train_loader:
        inst = binarize_instance_labels(inst_lbl.cpu().numpy())
        if len(inst) > 0:
            pos_fracs.append(inst.mean())

    if not pos_fracs:
        return None

    p_train = float(np.mean(pos_fracs))
    print(f'Baseline-Positiv-Anteil p_train = {p_train:.3f}')
    mlflow.log_metric('count_baseline_p_train', p_train)
    return p_train


# ══════════════════════════════════════════════════════════════════════════════
# Training and validation
# ══════════════════════════════════════════════════════════════════════════════

def _log_grayscale_metrics(model, epoch):
    """Log the learned RGB->gray weights and their similarity to reference conversions."""
    gray = getattr(model, "grayscale_layer", None)
    if not isinstance(gray, LearnedGrayscale):
        return

    w = gray.normalized_weights()               # tensor([r, g, b])
    mlflow.log_metrics({
        "gray_w_r": w[0].item(),
        "gray_w_g": w[1].item(),
        "gray_w_b": w[2].item(),
    }, step=epoch)

    # Cosine similarity to reference conversions -> "which index it's similar to"
    sims = {k: F.cosine_similarity(w, v, dim=0).item() for k, v in GRAYSCALE_REFERENCES.items()}
    mlflow.log_metrics({f"gray_cos_{k}": s for k, s in sims.items()}, step=epoch)


def train_one_epoch(ctx, epoch):
    """Run one training epoch over all bags and log the epoch losses to MLflow."""
    # ── Setup ─────────────────────────────────────────────────────────────────
    args, model, optimizer = ctx.args, ctx.model, ctx.optimizer
    model.train()

    # ── Epoch accumulators ────────────────────────────────────────────────────
    train_loss = 0.
    train_error = 0.
    train_bag_loss = 0.
    train_inst_loss = 0.

    # ── Main loop (one bag per step) ──────────────────────────────────────────
    for batch_idx, (patches, coords, label, count, instance_label) in enumerate(ctx.train_loader):
        bag_label = label
        if args.cuda:
            patches, bag_label = patches.cuda(), bag_label.cuda()

        patches = patches.squeeze(0)  # Drop the batch dimension, it is always 1

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        if ctx.is_clam:
            total_loss, _, bag_loss, inst_loss = model.calculate_objective(
                patches, bag_label, instance_eval=True, bag_weight=args.clam_bag_weight
            )
            train_loss += total_loss.item()
            train_bag_loss += bag_loss.item()
            train_inst_loss += inst_loss.item() if torch.is_tensor(inst_loss) else float(inst_loss)
            total_loss.backward()
        else:
            loss, _ = model.calculate_objective(patches, bag_label)
            train_loss += loss.item()
            loss.backward()

        error, _ = model.calculate_classification_error(patches, bag_label)
        train_error += error
        # backward pass
        # step
        optimizer.step()

    # ── Epoch summary ─────────────────────────────────────────────────────────
    train_loss /= len(ctx.train_loader)
    train_error /= len(ctx.train_loader)
    train_inst_loss /= len(ctx.train_loader)
    train_bag_loss /= len(ctx.train_loader)

    # ── MLflow logging ────────────────────────────────────────────────────────
    mlflow.log_metric('train_loss', train_loss, step=epoch)
    mlflow.log_metric('train_error', train_error, step=epoch)
    if ctx.is_clam:
        mlflow.log_metric('train_bag_loss', train_bag_loss, step=epoch)
        mlflow.log_metric('train_instance_loss', train_inst_loss, step=epoch)
        print('Epoch: {}, Loss: {:.4f} (bag {:.4f} / inst {:.4f}), Train error: {:.4f}'.format(
            epoch, train_loss, train_bag_loss, train_inst_loss, train_error))
    else:
        print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss, train_error))

    _log_grayscale_metrics(model, epoch)


def validate(ctx, epoch):
    """Evaluate on the validation split and log bag-level metrics to MLflow.

    In the final epoch (CLAM + --count_threshold_eval) the per-bag instance scores
    are additionally stored on the context for threshold calibration in test().
    """
    # ── Setup ─────────────────────────────────────────────────────────────────
    args, model = ctx.args, ctx.model
    model.eval()

    # ── Bag-level accumulators ────────────────────────────────────────────────
    val_loss = 0.
    val_error = 0.
    y_true, y_pred, y_prob = [], [], []

    # ── Instance scores/labels of the CLAM instance classifier (quantile stats) ─
    val_inst_scores, val_inst_labels = [], []

    # ── Collect calibration scores only in the last epoch ─────────────────────
    collect_val_scores = ((args.count_threshold_eval or args.calibrate_count_threshold)
                          and ctx.is_clam and epoch == args.epochs)
    if collect_val_scores:
        ctx.val_scores_per_bag.clear()
        ctx.val_true_counts.clear()
        ctx.val_pred_pos_per_bag.clear()

    # ── Main loop ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        for batch_idx, (patches, coords, label, count, instance_label) in enumerate(ctx.val_loader):
            if args.cuda:
                patches, bag_label = patches.cuda(), label.cuda()
            else:
                bag_label = label

            patches = patches.squeeze(0)

            if ctx.is_clam:
                logits, Y_prob_full, predicted_label, _, _ = model(patches)
                prob_pos = Y_prob_full[0, 1]                    # P(positive)
                loss = F.cross_entropy(logits, bag_label.long().view(-1))
                val_loss += loss.item()
                pred = predicted_label.view(-1).float()
                error = 1. - pred.eq(bag_label.float().view(-1)).cpu().float().mean().item()
                val_error += error
                y_prob.append(prob_pos.cpu().item())
                y_pred.append(pred.cpu().item())

                # Instance scores come from the instance classifier, NOT from attention
                _, inst_probs = model.count_positive_instances(patches.unsqueeze(0), threshold=0.5)
                inst_probs_np = inst_probs.cpu().numpy()

                inst_lbl = binarize_instance_labels(instance_label.cpu().numpy())
                m = min(len(inst_lbl), inst_probs_np.shape[0])
                val_inst_labels.extend(inst_lbl[:m].tolist())
                val_inst_scores.extend(inst_probs_np[:m].tolist())

                if collect_val_scores:
                    ctx.val_scores_per_bag.append(inst_probs_np)
                    ctx.val_true_counts.append(int(count.item()) if torch.is_tensor(count) else int(count))
                    ctx.val_pred_pos_per_bag.append(bool(pred.cpu().item() == 1))
            else:
                Y_prob, predicted_label, _ = model(patches)

                bag_label_f = bag_label.float()
                Y_prob_clamped = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
                loss = -1. * (bag_label_f * torch.log(Y_prob_clamped)
                              + (1. - bag_label_f) * torch.log(1. - Y_prob_clamped))
                val_loss += loss.item()
                error = 1. - predicted_label.eq(bag_label_f).cpu().float().mean().data.item()
                val_error += error

                y_pred.append(predicted_label.cpu().item())
                y_prob.append(Y_prob.cpu().item())

            y_true.append(bag_label.cpu().item())

    # ── Epoch summary ─────────────────────────────────────────────────────────
    val_loss /= len(ctx.val_loader)
    val_error /= len(ctx.val_loader)

    metrics = calculate_metrics(y_true, y_pred, y_prob)
    print('Epoch: {}, Val Loss: {:.4f}, Val Error: {:.4f}, Val AUC: {:.4f}\n'.format(
        epoch, val_loss, val_error, metrics['auc']))

    # ── MLflow logging ────────────────────────────────────────────────────────
    mlflow.log_metrics({
        "val_loss": val_loss,
        "val_error": val_error,
        "val_accuracy": metrics['accuracy'],
        "val_precision": metrics['precision'],
        "val_recall": metrics['recall'],
        "val_f1_score": metrics['f1_score'],
        "val_auc": metrics['auc'],
        "val_mae": metrics['mae'],
        "val_rmse": metrics['rmse'],
        "val_bias": metrics['bias'],
    }, step=epoch)

    # Quantiles of the instance scores (CLAM only, instance classifier scores)
    if ctx.is_clam and len(val_inst_labels) > 0:
        quantiles = calculate_score_quantiles(val_inst_scores, val_inst_labels)
        # Skip NaN entries (no positive resp. negative instances present)
        mlflow.log_metrics(
            {f'val_inst_{k}': v for k, v in quantiles.items() if not np.isnan(v)},
            step=epoch)


# ══════════════════════════════════════════════════════════════════════════════
# Testing
# ══════════════════════════════════════════════════════════════════════════════

def _calibrate_count_threshold(ctx):
    """Kalibriere den globalen Zaehl-Threshold auf dem Validierungssplit (CLAM).

    Muss VOR dem Test-Loop laufen, damit ``count_positive_instances`` dort bereits
    mit dem kalibrierten Wert zaehlt. Das Ergebnis wird am Modell hinterlegt
    (Buffer -> wandert ins state_dict) und auf dem Kontext gemerkt, damit
    ``_evaluate_count_thresholds`` nicht ein zweites Mal kalibriert.
    """
    args = ctx.args
    if not (args.calibrate_count_threshold and ctx.is_clam):
        return

    if not ctx.val_scores_per_bag:
        print("Threshold-Kalibrierung uebersprungen (keine Val-Scores gesammelt).")
        return

    val_gate = np.asarray(ctx.val_pred_pos_per_bag, dtype=bool)
    thr, val_bias, val_mae = CLAM.calibrate_threshold(
        ctx.val_scores_per_bag, ctx.val_true_counts, pred_pos=val_gate)

    ctx.count_threshold = thr
    ctx.model.set_count_threshold(thr)

    print(f'Zaehl-Threshold auf Val kalibriert: thr={thr:.2f} '
          f'(Val-Bias={val_bias:+.2f}, Val-MAE={val_mae:.2f})')
    mlflow.log_metrics({'count_calibrated_threshold': thr,
                        'count_calibrated_val_bias': val_bias,
                        'count_calibrated_val_mae': val_mae})


def _run_test_loop(ctx, results):
    """Forward every test bag once and collect all per-bag values.

    Also appends the per-bag predictions to the cross-seed ``results`` table.

    Returns:
        TestBuffers with the collected values (losses already averaged).
    """
    # ── Setup ─────────────────────────────────────────────────────────────────
    args, model = ctx.args, ctx.model
    model.eval()
    buf = TestBuffers()

    # ── Main loop ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        for batch_idx, (patches, coords, label, count, instance_label) in enumerate(ctx.test_loader):
            if args.cuda:
                patches, bag_label = patches.cuda(), label.cuda()
            else:
                bag_label = label
            patches = patches.squeeze(0)  # Drop the batch dimension, it is always 1

            if ctx.is_clam:
                logits, Y_prob_full, predicted_label, A_raw, _ = model(patches)
                prob_pos = Y_prob_full[0, 1]
                loss = F.cross_entropy(logits, bag_label.long().view(-1))
                buf.test_loss += loss.item()
                pred = predicted_label.view(-1).float()
                error = 1. - pred.eq(bag_label.float().view(-1)).cpu().float().mean().item()
                buf.test_error += error
                buf.y_prob.append(prob_pos.cpu().item())
                buf.y_pred.append(pred.cpu().item())

                # Instance scores come from the instance classifier, NOT from attention
                _, inst_probs = model.count_positive_instances(patches.unsqueeze(0), threshold=0.5)

                if args.count_threshold_eval:
                    buf.scores_per_bag.append(inst_probs.cpu().numpy())
                    buf.true_counts.append(int(count.item()) if torch.is_tensor(count) else int(count))
                    buf.pred_pos_per_bag.append(bool(pred.cpu().item() == 1))

                inst_lbl = binarize_instance_labels(instance_label.cpu().numpy())
                m = min(len(inst_lbl), inst_probs.shape[0])
                buf.patch_inst_labels.extend(inst_lbl[:m].tolist())
                buf.patch_inst_scores.extend(inst_probs[:m].cpu().numpy().tolist())

                if ctx.log_instance_scores:
                    buf.inst_score_agg.append(inst_probs[:m].cpu().numpy().tolist())
                    buf.inst_label_agg.append(inst_lbl[:m].tolist())
            else:
                # Single forward pass to avoid redindant computation
                Y_prob, predicted_label, attention_weights = model(patches)

                bag_label_f = bag_label.float()
                Y_prob_clamped = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
                loss = -1. * (bag_label_f * torch.log(Y_prob_clamped)
                              + (1. - bag_label_f) * torch.log(1. - Y_prob_clamped))
                buf.test_loss += loss.item()
                error = 1. - predicted_label.eq(bag_label_f).cpu().float().mean().data.item()
                buf.test_error += error

                buf.y_pred.append(predicted_label.cpu().item())
                buf.y_prob.append(Y_prob.cpu().item())

            buf.y_true.append(bag_label.cpu().item())

            # ── Per-bag rows of the cross-seed result table ────────────────────
            results["seeds"].append(ctx.seed)
            results["bag_ids"].append(batch_idx)
            results["truth"].append(bag_label.cpu().item())
            results["predicted"].append(predicted_label.cpu().item())

            if args.log_attention_weights and not ctx.is_clam:
                buf.attention_agg.append(attention_weights.cpu().numpy().tolist())

            # ── Naive counting (only for bags predicted positive) ─────────────
            if args.naive_counting:
                if predicted_label.cpu().item() == 1:
                    if ctx.is_clam:
                        # threshold=None -> model.count_threshold (ggf. auf Val kalibriert)
                        predicted_count, _ = model.count_positive_instances(patches.unsqueeze(0))
                        threshold = float(model.count_threshold)
                    else:
                        predicted_count, _, threshold = model.count_positive_instances(patches)
                else:
                    predicted_count = 0
                    threshold = 0
                buf.count_truth.append(count)
                buf.count_pred.append(predicted_count)
                results["count_threshold"].append(threshold)

                if predicted_label.cpu().item() == 1 and not ctx.is_clam:
                    flat_labels = instance_label.cpu().numpy().flatten().tolist()
                    buf.all_instance_labels.extend(flat_labels)
                    buf.all_attention_weights.extend(attention_weights.cpu().numpy().flatten().tolist())
                    buf.all_thresholds.extend([threshold] * len(flat_labels))

    # ── Averages over the split ───────────────────────────────────────────────
    buf.test_loss /= len(ctx.test_loader)
    buf.test_error /= len(ctx.test_loader)

    return buf


def _log_patch_level_metrics(ctx, buf, results, metrics):
    """Compute patch-level AUC/precision/recall and write them to MLflow and ``metrics``.

    Two sources are supported: thresholded attention weights (attention models)
    and the CLAM instance classifier scores.
    """
    args = ctx.args

    # ── Attention-based models: thresholded attention weights ─────────────────
    if (len(buf.all_instance_labels) > 0 and len(buf.all_attention_weights) > 0
            and "count_threshold" in results
            and len(results["count_threshold"]) > 0):
        instance_labels = binarize_instance_labels(buf.all_instance_labels)
        attention_converted = [1 if buf.all_attention_weights[i] > buf.all_thresholds[i] else 0
                               for i in range(len(buf.all_attention_weights))]

        conv_patch_auc = roc_auc_score(instance_labels, attention_converted)
        patch_auc = roc_auc_score(instance_labels, buf.all_attention_weights)
        patch_precision = precision_score(instance_labels, attention_converted, zero_division=0)
        patch_recall = recall_score(instance_labels, attention_converted, zero_division=0)

        mlflow.log_metrics({
            'patch_level_auc': patch_auc,
            'patch_level_precision': patch_precision,
            'patch_level_recall': patch_recall,
            'patch_level_auc_converted': conv_patch_auc,
        })
        metrics['patch_level_auc_converted'] = conv_patch_auc
        print(f'Patch-Level AUC (conv): {patch_auc:.4f} ({conv_patch_auc:.4f}), '
              f'Precision: {patch_precision:.4f} , Recall: {patch_recall:.4f}')

    # ── CLAM: scores of the instance classifier ───────────────────────────────
    elif ctx.is_clam and len(buf.patch_inst_labels) > 0:
        inst_labels_arr = np.array(buf.patch_inst_labels)
        inst_scores_arr = np.array(buf.patch_inst_scores)
        inst_preds = (inst_scores_arr >= 0.5).astype(int)

        if len(set(buf.patch_inst_labels)) > 1:
            patch_auc = roc_auc_score(inst_labels_arr, inst_scores_arr)
        else:
            patch_auc = 0.0
        patch_precision = precision_score(inst_labels_arr, inst_preds, zero_division=0)
        patch_recall = recall_score(inst_labels_arr, inst_preds, zero_division=0)

        mlflow.log_metrics({
            'patch_level_auc': patch_auc,
            'patch_level_precision': patch_precision,
            'patch_level_recall': patch_recall,
        })
        print(f'Patch-Level AUC (inst. clf.): {patch_auc:.4f}, '
              f'Precision: {patch_precision:.4f}, Recall: {patch_recall:.4f}')

    # ── No positive bags: nothing computable ──────────────────────────────────
    else:
        patch_auc = 0
        patch_precision = 0
        patch_recall = 0
        print('Patch-Level AUC/Precision/Recall: Could not be computed (no positive bags)')

    metrics['patch_level_auc'] = patch_auc
    metrics['patch_level_precision'] = patch_precision
    metrics['patch_level_recall'] = patch_recall


def _log_counting_metrics(ctx, buf, results, metrics):
    """Compute naive counting metrics, log them and store the truth/prediction scatter plot."""
    if not (ctx.args.naive_counting and len(buf.count_pred) > 0):
        return

    # ── Metrics ───────────────────────────────────────────────────────────────
    counting_metrics = calculate_counting_metrics(buf.count_truth, buf.count_pred)
    metrics['counting_accuracy'] = counting_metrics['counting_accuracy']
    metrics['counting_mae'] = counting_metrics['counting_mae']
    metrics['counting_rmse'] = counting_metrics['counting_rmse']
    mlflow.log_metric('counting_accuracy', counting_metrics['counting_accuracy'])

    # ── Unwrap tensors so the values can be logged/plotted ────────────────────
    clean_truth = [int(c[0].item()) if isinstance(c, list) and torch.is_tensor(c[0])
                   else int(c.item()) if torch.is_tensor(c) else int(c) for c in buf.count_truth]
    clean_pred = [int(p.item()) if torch.is_tensor(p) else int(p) for p in buf.count_pred]

    results["count_truth"].extend(clean_truth)
    results["count_pred"].extend(clean_pred)

    # ── Per-seed scatter plot: truth vs. prediction ───────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(clean_truth, clean_pred, alpha=0.6, edgecolors='w')

    max_val = max(max(clean_truth), max(clean_pred))
    ax.plot([0, max_val], [0, max_val], 'r--')

    ax.set_xlabel("True Count")
    ax.set_ylabel("Predicted Count")
    ax.set_title("Count Truth vs. Prediction")

    mlflow.log_figure(fig, "counting_plot.png")
    plt.close(fig)

    mlflow.log_metrics({"count_mae": counting_metrics['counting_mae'],
                        "count_rmse": counting_metrics['counting_rmse']})
    print('Counting Accuracy: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}'.format(
        counting_metrics['counting_accuracy'], counting_metrics['counting_mae'],
        counting_metrics['counting_rmse']))


def _evaluate_count_thresholds(ctx, buf, metrics):
    """Compare counting-threshold strategies on the test split (CLAM only).

    All strategies are bag-gated: bags classified as negative count 0. That keeps
    the comparison fair against naive (also gated) counting and matches the
    zero-inflated count distribution.

    Strategies: global sweep, per-bag Otsu, validation-calibrated global threshold,
    soft count and the trivial p_train * N baseline.
    """
    args = ctx.args
    if not (args.count_threshold_eval and ctx.is_clam and len(buf.scores_per_bag) > 0):
        return

    gate = np.asarray(buf.pred_pos_per_bag, dtype=bool)

    # ── 1) Global sweep -- bias/MAE curve over thresholds (gated) ─────────────
    print("\n--- Counting-Threshold-Sweep (Test, bag-gegatet) ---")
    for thr in COUNT_SWEEP_THRESHOLDS:
        bias, mae = CLAM.counting_scores_per_bag(buf.scores_per_bag, buf.true_counts, thr, pred_pos=gate)
        print(f"  thr={thr:.2f}  Bias={bias:+.2f}  MAE={mae:.2f}")
        mlflow.log_metric("count_sweep_bias", bias, step=int(thr * 100))
        mlflow.log_metric("count_sweep_mae", mae, step=int(thr * 100))

    # ── 2) Per-bag Otsu (gated) ───────────────────────────────────────────────
    otsu_bias, otsu_mae = CLAM.counting_scores_otsu(buf.scores_per_bag, buf.true_counts, pred_pos=gate)
    print(f"  Otsu (per-bag)  Bias={otsu_bias:+.2f}  MAE={otsu_mae:.2f}")
    mlflow.log_metric("count_otsu_bias", otsu_bias)
    mlflow.log_metric("count_otsu_mae", otsu_mae)
    metrics['count_otsu_mae'] = otsu_mae

    # ── 3) Global threshold, bias-calibrated on validation (val + test gated) ─
    if ctx.count_threshold is not None or len(ctx.val_scores_per_bag) > 0:
        if ctx.count_threshold is not None:
            # Bereits vor dem Test-Loop kalibriert (--calibrate_count_threshold)
            cal_thr = ctx.count_threshold
        else:
            val_gate = np.asarray(ctx.val_pred_pos_per_bag, dtype=bool)
            cal_thr, _cal_bias_val, _cal_mae_val = CLAM.calibrate_threshold(
                ctx.val_scores_per_bag, ctx.val_true_counts, pred_pos=val_gate)
        test_bias, test_mae = CLAM.counting_scores_per_bag(
            buf.scores_per_bag, buf.true_counts, cal_thr, pred_pos=gate)
        print(f"  Kalibriert (thr={cal_thr:.2f} aus Val)  Test-Bias={test_bias:+.2f}  Test-MAE={test_mae:.2f}")
        mlflow.log_metric("count_calibrated_threshold", cal_thr)
        mlflow.log_metric("count_calibrated_bias", test_bias)
        mlflow.log_metric("count_calibrated_mae", test_mae)
        metrics['count_calibrated_threshold'] = cal_thr
        metrics['count_calibrated_mae'] = test_mae
    else:
        print("  Kalibrierung uebersprungen (keine Val-Scores gesammelt).")

    # ── 4) Soft count: sum of instance probabilities (gated, threshold-free) ──
    if args.soft_counting:
        soft_bias, soft_mae = CLAM.counting_scores_soft(buf.scores_per_bag, buf.true_counts, pred_pos=gate)
        print(f"  Soft-Count (Sum P, gegatet)  Bias={soft_bias:+.2f}  MAE={soft_mae:.2f}")
        mlflow.log_metric("count_soft_bias", soft_bias)
        mlflow.log_metric("count_soft_mae", soft_mae)
        metrics['count_soft_mae'] = soft_mae

    # ── 5) Trivial baseline estimator: fixed fraction p_train * N (ungated) ───
    if ctx.p_train is not None:
        pred = np.array([ctx.p_train * len(s) for s in buf.scores_per_bag])
        true = np.array(buf.true_counts)
        base_bias = float((pred - true).mean())
        base_mae = float(np.abs(pred - true).mean())
        print(f"  Baseline (p={ctx.p_train:.2f} * N)  Bias={base_bias:+.2f}  MAE={base_mae:.2f}")
        mlflow.log_metric("count_baseline_bias", base_bias)
        mlflow.log_metric("count_baseline_mae", base_mae)
        metrics['count_baseline_mae'] = base_mae


def test(ctx, results):
    """Evaluate the trained model on the test split and log all metrics/artifacts.

    Returns:
        (metrics, count_truth, count_pred) for the current seed.
    """
    # ── Zaehl-Threshold aus dem Val-Split (vor dem Test-Loop!) ────────────────
    _calibrate_count_threshold(ctx)

    # ── Forward pass over the test split ──────────────────────────────────────
    buf = _run_test_loop(ctx, results)

    print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(buf.test_loss, buf.test_error))

    # ── Bag-level metrics ─────────────────────────────────────────────────────
    metrics = calculate_metrics(buf.y_true, buf.y_pred, buf.y_prob)
    print('Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, '
          'F1-Score: {:.4f}, AUC: {:.4f}'.format(
              metrics['accuracy'], metrics['precision'], metrics['recall'],
              metrics['f1_score'], metrics['auc']))

    # ── Patch-level metrics ───────────────────────────────────────────────────
    _log_patch_level_metrics(ctx, buf, results, metrics)

    mlflow.log_metrics({"test_loss": buf.test_loss,
                        "test_error": buf.test_error,
                        "accuracy": metrics['accuracy'],
                        "precision": metrics['precision'],
                        "recall": metrics['recall'],
                        "f1_score": metrics['f1_score'],
                        "auc": metrics['auc'],
                        "mae": metrics['mae'],
                        "rmse": metrics['rmse'],
                        "bias": metrics['bias']})

    metrics['test_loss'] = buf.test_loss
    metrics['test_error'] = buf.test_error

    # ── Counting metrics and threshold strategies ─────────────────────────────
    _log_counting_metrics(ctx, buf, results, metrics)
    _evaluate_count_thresholds(ctx, buf, metrics)

    # ── Artifact aggregation for the parent run ───────────────────────────────
    if ctx.args.log_attention_weights and buf.attention_agg:
        results["attention_weights"].extend(buf.attention_agg)

    if ctx.log_instance_scores and buf.inst_score_agg:
        results["instance_scores"].extend(buf.inst_score_agg)
        results["instance_labels"].extend(buf.inst_label_agg)

    return metrics, buf.count_truth, buf.count_pred


# ══════════════════════════════════════════════════════════════════════════════
# Feature visualization and cross-seed aggregation
# ══════════════════════════════════════════════════════════════════════════════

def log_seed_feature_plot(ctx, all_feature_data):
    """Collect test-set features for the current seed, log the plot in the child run
    and keep the raw arrays for the aggregated parent plot."""
    model = ctx.model

    if not hasattr(model, 'extract_features'):
        print("Warning: Modell hat keine extract_features()-Methode. "
              "Bitte model.py aktualisieren. Visualisierung übersprungen.")
        return

    try:
        print(f"\nSammle Features für Visualisierung...")
        H, A, bag_lbls, inst_lbls, bag_ids = vf.collect_features_from_loader(
            model, ctx.test_loader, device=ctx.device
        )
        H_2d = vf.reduce_dimensions(H, "umap")
        fig = vf.plot_per_seed(
            H_2d, A, bag_lbls, inst_lbls,
            title_suffix=f"Seed {ctx.seed}"
        )
        mlflow.log_figure(fig, "feature_visualization.png")
        plt.close(fig)
        print(f"  Feature-Plot in Child Run geloggt (seed={ctx.seed}).")

        # Keep the raw data for the aggregated parent plot
        all_feature_data.append({
            "H":               H,
            "A":               A,
            "bag_labels":      bag_lbls,
            "instance_labels": inst_lbls,
            "seed_ids":        np.full(len(H), ctx.seed, dtype=int),
        })
    except Exception as viz_err:
        print(f"Feature-Visualisierung für Seed {ctx.seed} fehlgeschlagen: {viz_err}")


def log_aggregated_metrics(all_metrics):
    """Log mean and std of every metric key across all seeds to the parent run."""
    if not all_metrics:
        return

    print("\nAggregating metrics across seeds...")
    metrics_keys = all_metrics[0].keys()

    for key in metrics_keys:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            mean_value = np.mean(values)
            std_value = np.std(values)
            mlflow.log_metric(f'{key}_mean', mean_value)
            mlflow.log_metric(f'{key}_std', std_value)
            print(f"{key}: Mean = {mean_value:.4f}, Std = {std_value:.4f}")


def log_aggregated_counting_plot(results):
    """Log one scatter plot of true vs. predicted counts, colored per seed."""
    if not (results.get("count_truth") and results.get("count_pred")):
        return

    print("Logging aggregated counting artifacts...")

    fig, ax = plt.subplots(figsize=(8, 8))
    unique_seeds = sorted(set(results["seeds"]))

    for i, seed in enumerate(unique_seeds):
        seed_truth = [results["count_truth"][j] for j, s in enumerate(results["seeds"]) if s == seed]
        seed_pred = [results["count_pred"][j] for j, s in enumerate(results["seeds"]) if s == seed]

        # Pick the matching color (modulo in case there are more seeds than colors)
        color = SEED_PLOT_COLORS[i % len(SEED_PLOT_COLORS)]

        ax.scatter(
            seed_truth, seed_pred,
            color=color,
            label=f"Seed {seed}",
            alpha=0.6, edgecolors='w'
        )

    max_val = max(max(results["count_truth"]), max(results["count_pred"]))
    ax.plot([0, max_val], [0, max_val], 'r--', label="Perfect Prediction")

    ax.set_xlabel("True Count")
    ax.set_ylabel("Predicted Count")
    ax.set_title("Aggregated Count: Truth vs. Prediction (All Seeds)")
    ax.legend()

    mlflow.log_figure(fig, "aggregated_counting_plot.png")
    plt.close(fig)


def log_aggregated_feature_plot(all_feature_data):
    """Log a single UMAP plot over the features of all seeds in the parent run."""
    if len(all_feature_data) == 0:
        return

    try:
        print("\nErstelle aggregierten Feature-Plot über alle Seeds...")
        H_agg = np.concatenate([d["H"] for d in all_feature_data])
        A_agg = np.concatenate([d["A"] for d in all_feature_data])
        bl_agg = np.concatenate([d["bag_labels"] for d in all_feature_data])
        il_agg = np.concatenate([d["instance_labels"] for d in all_feature_data])
        sid_agg = np.concatenate([d["seed_ids"] for d in all_feature_data])

        print(f"  Gesamt: {len(H_agg)} Instanzen aus {len(all_feature_data)} Seed(s).")
        H_2d_agg = vf.reduce_dimensions(H_agg, "umap")

        if len(all_feature_data) > 1:
            # Multiple seeds -> 4-panel plot including seed membership
            fig = vf.plot_aggregated(H_2d_agg, A_agg, bl_agg, il_agg, sid_agg)
        else:
            # Single seed -> same 3-panel plot as in the child run,
            # but explicitly logged in the parent run
            fig = vf.plot_per_seed(
                H_2d_agg, A_agg, bl_agg, il_agg,
                title_suffix=f"Seed {int(sid_agg[0])} (Parent)"
            )
        mlflow.log_figure(fig, "feature_visualization_aggregated.png")

        plt.close(fig)
        print("  Aggregierter Feature-Plot in Parent Run geloggt.")
    except Exception as viz_err:
        print(f"Aggregierte Feature-Visualisierung fehlgeschlagen: {viz_err}")


# ══════════════════════════════════════════════════════════════════════════════
# Single seed run
# ══════════════════════════════════════════════════════════════════════════════

def run_seed(args, seed, results, all_metrics, all_feature_data, log_instance_scores):
    """Train, validate and test one seed inside a nested MLflow child run.

    Returns:
        The MLflow tags describing the instantiated model.
    """
    print(f'\n{"="*30}\nRunning with seed: {seed}\n{"="*30}\n')

    # ── Seeding ───────────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
        print('\nGPU is ON!')

    # Start nested child run with current seed
    with mlflow.start_run(run_name=f"seed_{seed}", nested=True) as child_run:
        mlflow.log_param('current_seed', seed)

        # ── Data, model, optimizer ────────────────────────────────────────────
        train_loader, val_loader, test_loader = build_dataloaders(args)
        model, model_tags = build_model(args)

        if args.pretrained_backbone:
            load_pretrained_backbone(model, args.pretrained_backbone)

        if args.cuda:
            model.cuda()

        # ── Gelernte Gewichte laden (Weitertrainieren oder --eval_only) ───────
        if args.load_model:
            load_model_state(model, args.load_model, "cuda" if args.cuda else "cpu")

        optimizer = build_optimizer(model, args)

        ctx = RunContext(args=args, seed=seed,
                         model=model, optimizer=optimizer,
                         train_loader=train_loader,
                         val_loader=val_loader,
                         test_loader=test_loader,
                         log_instance_scores=log_instance_scores)

        # ── Baseline positive fraction for the counting evaluation (CLAM) ─────
        if args.count_threshold_eval and ctx.is_clam:
            ctx.p_train = compute_train_positive_fraction(train_loader)

        # ── Training ──────────────────────────────────────────────────────────
        if args.eval_only:
            print('--eval_only: Training wird uebersprungen.')
            # Val-Scores fuer die Threshold-Kalibrierung trotzdem einsammeln
            if (args.count_threshold_eval or args.calibrate_count_threshold) and ctx.is_clam:
                validate(ctx, args.epochs)
        else:
            print('Starting training!')
            for epoch in range(1, args.epochs + 1):
                train_one_epoch(ctx, epoch)
                validate(ctx, epoch)

        # ── Testing ───────────────────────────────────────────────────────────
        print('Starting testing!')
        metrics, seed_truth, seed_pred = test(ctx, results)

        all_metrics.append(metrics)

        # ── State_dict speichern ──────────────────────────────────────────────
        if args.save_model and not args.eval_only:
            save_model_state(model, args, seed)
        mlflow.pytorch.log_model(model, "model")  # Log the model to MLflow

        # ── Feature visualization (child run / per seed) ──────────────────────
        if args.visualize_features:
            log_seed_feature_plot(ctx, all_feature_data)

    return model_tags


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Configuration ─────────────────────────────────────────────────────────
    args, config_path = parse_args()
    config_filename = save_run_config(args) if config_path is None else None

    if args.exp_name:
        mlflow.set_experiment(args.exp_name)

    # ── Derived logging flags ─────────────────────────────────────────────────
    # Instance scores of the CLAM instance classifier are only available for CLAM
    log_instance_scores = args.log_instance_scores and args.model == 'clam'
    if args.log_instance_scores and not log_instance_scores:
        print("Warning: --log_instance_scores ist nur fuer --model clam verfuegbar. Wird ignoriert.")

    default_run_name = (f"{args.model}_{args.dataset}_lr{args.lr}_reg{args.reg}"
                        f"_ep{args.epochs}_attention_activation{args.attention_activation}")

    # ──────────────────────────────────────────────────────────────────────────
    # Parent MLflow run: holds parameters and artifacts common to all seeds
    # ──────────────────────────────────────────────────────────────────────────
    with mlflow.start_run(run_name=args.run_name if args.run_name else default_run_name) as parent_run:
        mlflow.log_params(vars(args))
        mlflow.log_artifact(config_path if config_path is not None else config_filename)

        mlflow.set_tags({
            "model": args.model,
            "dataset": args.dataset,
            "num_seeds": len(args.seeds),
            "status": "in_progress"
        })

        # ── Cross-seed accumulators ───────────────────────────────────────────
        model_tags = {}
        all_metrics = []
        all_feature_data = []   # H, A, bag_lbls, inst_lbls and seed_ids of every seed
        results = init_results_container(args, log_instance_scores)

        try:
            # ── Iterate over each seed ────────────────────────────────────────
            for seed in args.seeds:
                model_tags = run_seed(args, seed, results, all_metrics,
                                      all_feature_data, log_instance_scores)

            # ── Logging in parent run after all seeds have been processed ─────
            mlflow.log_table(results, artifact_file="aggregated_run_results.json")
            mlflow.set_tags(model_tags)

            log_aggregated_metrics(all_metrics)
            log_aggregated_counting_plot(results)

            if args.visualize_features:
                log_aggregated_feature_plot(all_feature_data)

        except Exception as e:
            print(f"An error occurred: {e}")
            mlflow.log_param('error_message', str(e))
            mlflow.set_tags({"status": "failed"})
        else:
            mlflow.set_tags({"status": "completed"})


if __name__ == '__main__':
    main()
