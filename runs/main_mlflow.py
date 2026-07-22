import sys
import os

from models.learned_grayscale import LearnedGrayscale

# Fügt das Stammverzeichnis des Projekts zum Python-Pfad hinzu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import torch
import torch.utils.data as data_utils
import mlflow
import mlflow.pytorch
import mlflow.data as mlfdata
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import torch.nn.functional as F

from data.data_management.dataset_manager import DatasetReader
from eval.scripts.metrics import calculate_metrics, calculate_counting_metrics
from models.model import Attention, AttentionBatchNorm, AttentionDropout, AttentionThirdConv, GatedAttention
from models.fpn_mil_model import FPNMIL
import visualize_features as vf


# Get arguments from command line
parser = argparse.ArgumentParser(description='Testing Atteintion MIL models on datasets loaded from H5 files.')

#Base parameters
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
# Model parameters
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
parser.add_argument('--attention_activation', type=str, default='softmax', choices=['sigmoid', 'min_max', 'softmax', 'sparsemax', "softmax_temperature", "entmax"],
                    help='activation function for attention weights (default: softmax)')
parser.add_argument('--rgb', action='store_true', default=False,
                    help='use RGB input instead of grayscale')
parser.add_argument('--grayscale', action='store_true', default=False,
                    help='use learned grayscale conversion for RGB input (rgb and grayscale are exclusive)')
#FPN parameters
parser.add_argument('--model_dx', type=int, default=256,
                    help='FPN-MIL: shared FPN channel dimension d_x (default: 256)')
parser.add_argument('--model_base_channels', type=int, default=32,
                    help='FPN-MIL: base channel count of the backbone (default: 32)')
#CLAM parameters
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
parser.add_argument('--soft_counting', action='store_true', default=False,
                    help='CLAM: zusaetzlicher Soft-Count (Summe der Instanz-Wahrscheinlichkeiten, bag-gegatet) in count_threshold_eval')
#Data parameters
parser.add_argument('--dataset', type=str, default='mnist_bags', metavar='H5', 
                    help='path to H5 file containing the dataset (default: mnist_bags.h5)')
parser.add_argument('--path', type=str, default='../data/datasets/bags/mnist_bags.h5', metavar='H5',
                    help='path to H5 file containing the dataset (default: mnist_bags.h5)')
# MLflow parameters
parser.add_argument('--exp_name', type=str, default=None, metavar='EXP',
                    help='name of the MLflow experiment (default: default)')
parser.add_argument('--run_name', type=str, default=None, metavar='RUN',
                    help='name of the MLflow run (default: None)')
parser.add_argument('--log_attention_weights', action='store_true', default=False,
                    help='log attention weights as artifact in MLflow')
parser.add_argument('--visualize_features', action='store_true', default=False,
                    help='visualize extracted features using UMAP and log the plot to MLflow')


config_parser = argparse.ArgumentParser(description='Config file parser', add_help=False)
config_parser.add_argument('--config', type=str, default=None, metavar='CONFIG')
config_args, _ = config_parser.parse_known_args()

# Read configuration from YAML file if provided
if config_args.config is not None:
    if os.path.exists(config_args.config):
        print(f"Loading configuration from {config_args.config}")
        with open(config_args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        # Update default arguments with values from YAML config
        parser.set_defaults(**yaml_config)


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if config_args.config is None:
    print(f"No config file provided. Using default arguments and command line overrides.")
    args_dict = vars(args)
    config_filename = f"configs/{args.model}_{args.dataset}_lr{args.lr}_reg{args.reg}_ep{args.epochs}_attention_activation{args.attention_activation}_config.yaml"
    with open(config_filename, 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False)
    print(f"Run configuration saved to {config_filename}")

if args.exp_name:
    mlflow.set_experiment(args.exp_name)
# else:
#     experiment_name = f"{args.model}_{args.dataset}"
#     mlflow.set_experiment(experiment_name)

# Starting partent MLflow run to log parameters and artifacts common to all seeds

with mlflow.start_run(run_name=args.run_name if args.run_name else f"{args.model}_{args.dataset}_lr{args.lr}_reg{args.reg}_ep{args.epochs}_attention_activation{args.attention_activation}") as parent_run:
    mlflow.log_params(vars(args))
    if config_args.config is not None:
        mlflow.log_artifact(config_args.config)
    if config_args.config is None:
        mlflow.log_artifact(config_filename)

    # mlflow.log_input(
    #     mlfdata.from_path(
    #         path=args.path,
    #         name=args.dataset,
    #     )
    # )
    
    mlflow.set_tags({
        "model": args.model,
        "dataset": args.dataset,
        "num_seeds": len(args.seeds),
        #"hyperparams": "baseline",
        #"comparison_group": "v1.0",
        "status": "in_progress"
    })

    model_tags = {}
    all_metrics = []
    clean_truth, clean_pred = [], []

    all_runs_results = {
        "seeds": [],
        "bag_ids": [],
        "truth": [],
        "predicted": []
    }
    if args.naive_counting:
        all_runs_results["count_truth"] = []
        all_runs_results["count_pred"] = []
        all_runs_results["count_threshold"] = []

    if args.log_attention_weights:
        all_runs_results["attention_weights"] = []

    all_feature_data = []  # Liste zum Speichern von H, A, bag_lbls, inst_lbls und seed_ids für alle Seeds

    # Iterate over each seed
    try:
        for seed in args.seeds:
            print(f'\n{"="*30}\nRunning with seed: {seed}\n{"="*30}\n')

            torch.manual_seed(seed)
            if args.cuda:
                torch.cuda.manual_seed(seed)
                print('\nGPU is ON!')
            
            # Start nested child run with current seed
            with mlflow.start_run(run_name=f"seed_{seed}", nested=True) as child_run:
                mlflow.log_param('current_seed', seed)
                # Load dataset using DatasetReader
                print('Load Train and Test Set')
                loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
                train_dataset = data_utils.DataLoader(DatasetReader(args.path, dataset_name=args.dataset, split='train'),
                                                    batch_size=1, 
                                                    shuffle=True,
                                                    **loader_kwargs)
                val_dataset = data_utils.DataLoader(DatasetReader(args.path, dataset_name=args.dataset, split='validation'),
                                                    batch_size=1,
                                                    shuffle=False,
                                                    **loader_kwargs)
                test_dataset = data_utils.DataLoader(DatasetReader(args.path, dataset_name=args.dataset, split='test'),
                                                    batch_size=1,
                                                    shuffle=False,
                                                    **loader_kwargs)
                

                print('Initialize model')
                if args.model == 'attention':
                    model = Attention(M=args.model_M, L=args.model_L, num_maps=args.model_num_maps, kernel_size=args.model_kernel_size, pool_size=args.model_pool_size, in_channels=3 if args.rgb else 1, grayscaling=args.grayscaling, attention_activation=args.attention_activation)
                    model_tags = {
                        "model_M": model.M,
                        "model_L": model.L,
                        "model_pool_size": model.pool_size,
                        "model_num_maps": model.num_maps,
                        "model_kernel_size": model.kernel_size,
                        "model_attention_branches": model.ATTENTION_BRANCHES,
                        "model_architecture": "Conv2d(1->20->50) -> FC(800->M->L)"
                    }
                elif args.model == 'gated_attention':
                    model = GatedAttention(M=args.model_M, L=args.model_L, num_maps=args.model_num_maps, kernel_size=args.model_kernel_size, pool_size=args.model_pool_size, in_channels=3 if args.rgb else 1, attention_activation=args.attention_activation)
                    model_tags = {
                        "model_M": model.M,
                        "model_L": model.L,
                        "model_pool_size": model.pool_size,
                        "model_num_maps": model.num_maps,
                        "model_kernel_size": model.kernel_size,
                        "model_attention_branches": model.ATTENTION_BRANCHES,
                        "model_architecture": "GatedAttention: V(Tanh) + U(Sigmoid) + w"
                    }
                elif args.model == 'attention_batchnorm':
                    model = AttentionBatchNorm(M=args.model_M, L=args.model_L, num_maps=args.model_num_maps, kernel_size=args.model_kernel_size, pool_size=args.model_pool_size, in_channels=3 if args.rgb else 1, attention_activation=args.attention_activation)
                    model_tags = {
                        "model_M": model.M,
                        "model_L": model.L,
                        "model_pool_size": model.pool_size,
                        "model_num_maps": model.num_maps,
                        "model_kernel_size": model.kernel_size,
                        "model_attention_branches": model.ATTENTION_BRANCHES,
                        "model_architecture": "AttentionBatchNorm: Conv2d(1->20->50) + BatchNorm -> FC(800->M->L)"
                    } 
                elif args.model == 'attention_third_conv':
                    model = AttentionThirdConv(M=args.model_M, L=args.model_L, num_maps=args.model_num_maps, kernel_size=args.model_kernel_size, pool_size=args.model_pool_size, in_channels=3 if args.rgb else 1, attention_activation=args.attention_activation)
                    model_tags = {
                        "model_M": model.M,
                        "model_L": model.L,
                        "model_pool_size": model.pool_size,
                        "model_num_maps": model.num_maps,
                        "model_kernel_size": model.kernel_size,
                        "model_attention_branches": model.ATTENTION_BRANCHES,
                        "model_architecture": "AttentionThirdConv: Conv2d(1->20->50) -> FC(800->M->L)"
                    }
                elif args.model == 'attention_dropout':
                    model = AttentionDropout(M=args.model_M, L=args.model_L, num_maps=args.model_num_maps, kernel_size=args.model_kernel_size, pool_size=args.model_pool_size, in_channels=3 if args.rgb else 1, attention_activation=args.attention_activation)
                    model_tags = {
                        "model_M": model.M,
                        "model_L": model.L,
                        "model_pool_size": model.pool_size,
                        "model_num_maps": model.num_maps,
                        "model_kernel_size": model.kernel_size,
                        "model_attention_branches": model.ATTENTION_BRANCHES,
                        "model_architecture": "AttentionDropout: Conv2d(1->20->50) -> FC(800->M->L) + Dropout(p=0.25)"
                    }
                elif args.model == 'fpn_mil':
                    model = FPNMIL(in_channels=3 if args.rgb else 1,
                                   base_channels=args.model_base_channels,
                                   num_scales=args.model_num_scales,
                                   d_x=args.model_dx, d=args.model_M, L=args.model_L,
                                   kernel_size=args.model_kernel_size, gated=True)
                    model_tags = {
                        "model_M": args.model_M,
                        "model_L": args.model_L,
                        "model_kernel_size": args.model_kernel_size,
                        "model_num_scales": args.model_num_scales,
                        "model_dx": args.model_dx,
                        "model_base_channels": args.model_base_channels,
                        "model_architecture": f"FPN-MIL: {args.model_num_scales} scales, d_x={args.model_dx}, gated AbMIL + multi-scale aggregator"
                    }
                elif args.model == 'clam':
                    from models.clam_model import CLAM
                    model = CLAM(M=args.model_M, L=args.model_L, num_maps=args.model_num_maps,
                                kernel_size=args.model_kernel_size, pool_size=args.model_pool_size,
                                in_channels=3 if args.rgb else 1,
                                k_sample=args.clam_k_sample, pseudo_threshold=args.clam_pseudo_threshold, dropout=0.25, 
                                grayscaling=args.grayscale,
                                pseudo_quantile_pos=args.clam_pseudo_quantile_pos,
                                pseudo_quantile_neg=args.clam_pseudo_quantile_neg)
                    model_tags = {
                        "model_M": model.M,
                        "model_L": model.L,
                        "model_pool_size": model.pool_size,
                        "model_num_maps": model.num_maps,
                        "model_kernel_size": model.kernel_size,
                        "model_k_sample": model.k_sample,
                        "model_pseudo_threshold": model.pseudo_threshold,
                        "model_pseudo_quantile_pos": model.pseudo_quantile_pos,
                        "model_pseudo_quantile_neg": model.pseudo_quantile_neg,
                        "model_architecture": "CLAM_SB (gated attn + instance classifier)"
                    }
            

                if args.cuda:
                    model.cuda()

                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)

                # ── Puffer fuer Counting-Threshold-Auswertung (nur CLAM) ──────────────
                val_scores_per_bag = []   # Instanz-Scores je Val-Bag (letzte Epoche)
                val_true_counts = []      # wahre Counts je Val-Bag
                val_pred_pos_per_bag = [] # Bag-Vorhersage je Val-Bag (fuer Gating der Kalibrierung)
                p_train = None            # Positiv-Anteil aus dem Trainingsset (Baseline)

                if args.count_threshold_eval and args.model == 'clam':
                    print('Bestimme Baseline-Positiv-Anteil p_train ueber das Trainingsset...')
                    pos_fracs = []
                    for _p, _c, _lbl, _cnt, inst_lbl in train_dataset:
                        inst = inst_lbl.cpu().numpy().flatten()
                        if len(set(inst.tolist())) > 2:      # MNIST -> binaer
                            inst = (inst == 9).astype(int)
                        if len(inst) > 0:
                            pos_fracs.append(inst.mean())
                    if pos_fracs:
                        p_train = float(np.mean(pos_fracs))
                        print(f'Baseline-Positiv-Anteil p_train = {p_train:.3f}')
                        mlflow.log_metric('count_baseline_p_train', p_train)

                def train(epoch):
                    model.train()
                    train_loss = 0.
                    train_error = 0.
                    train_bag_loss = 0.
                    train_inst_loss = 0.
                    for batch_idx, (patches, coords, label, count, instance_label) in enumerate(train_dataset):
                        bag_label = label
                        if args.cuda:
                            patches, bag_label = patches.cuda(), bag_label.cuda()

                        patches = patches.squeeze(0)  # Entfernt die Batch-Dimension, da sie 1 ist

                        # reset gradients
                        optimizer.zero_grad()
                        # calculate loss and metrics
                        if args.model == 'clam':
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

                    # calculate loss and error for epoch
                    train_loss /= len(train_dataset)
                    train_error /= len(train_dataset)

                    train_inst_loss /= len(train_dataset)
                    train_bag_loss /= len(train_dataset)

                    # Log training loss and error to MLflow
                    mlflow.log_metric('train_loss', train_loss, step=epoch)
                    mlflow.log_metric('train_error', train_error, step=epoch)
                    if args.model == 'clam':
                        mlflow.log_metric('train_bag_loss', train_bag_loss , step=epoch)
                        mlflow.log_metric('train_instance_loss', train_inst_loss , step=epoch)
                        print('Epoch: {}, Loss: {:.4f} (bag {:.4f} / inst {:.4f}), Train error: {:.4f}'.format(
                            epoch, train_loss, train_bag_loss, train_inst_loss, train_error))
                    else:
                        print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss, train_error))
                    
                    gray = getattr(model, "grayscale_layer", None)
                    if isinstance(gray, LearnedGrayscale):
                        w = gray.normalized_weights()               # tensor([r, g, b])
                        mlflow.log_metrics({
                            "gray_w_r": w[0].item(),
                            "gray_w_g": w[1].item(),
                            "gray_w_b": w[2].item(),
                        }, step=epoch)

                    # cosine similarity to reference conversions -> "which index it's similar to"
                    refs = {
                        "bt601":  torch.tensor([0.299, 0.587, 0.114]),
                        "bt709":  torch.tensor([0.2126, 0.7152, 0.0722]),
                        "R":      torch.tensor([1., 0., 0.]),
                        "G":      torch.tensor([0., 1., 0.]),
                        "B":      torch.tensor([0., 0., 1.]),
                        "mean":   torch.tensor([1/3, 1/3, 1/3]),
                    }
                    sims = {k: F.cosine_similarity(w, v, dim=0).item() for k, v in refs.items()}
                    mlflow.log_metrics({f"gray_cos_{k}": s for k, s in sims.items()}, step=epoch)


                def validate(epoch):
                    model.eval()
                    val_loss = 0.
                    val_error = 0.
                    y_true, y_pred, y_prob = [], [], []

                    # Instanz-Scores fuer die Val-Kalibrierung nur in der letzten Epoche sammeln
                    collect_val_scores = args.count_threshold_eval and args.model == 'clam' and epoch == args.epochs
                    if collect_val_scores:
                        val_scores_per_bag.clear()
                        val_true_counts.clear()
                        val_pred_pos_per_bag.clear()

                    with torch.no_grad():
                        for batch_idx, (patches, coords, label, count, instance_label) in enumerate(val_dataset):
                            if args.cuda:
                                patches, bag_label = patches.cuda(), label.cuda()
                            else:
                                bag_label = label

                            patches = patches.squeeze(0)

                            if args.model == 'clam':
                                logits, Y_prob_full, predicted_label, _, _ = model(patches)
                                prob_pos = Y_prob_full[0, 1]                    # P(positiv)
                                loss = F.cross_entropy(logits, bag_label.long().view(-1))
                                val_loss += loss.item()
                                pred = predicted_label.view(-1).float()
                                error = 1. - pred.eq(bag_label.float().view(-1)).cpu().float().mean().item()
                                val_error += error
                                y_prob.append(prob_pos.cpu().item())
                                y_pred.append(pred.cpu().item())

                                if collect_val_scores:
                                    _, inst_probs = model.count_positive_instances(patches.unsqueeze(0), threshold=0.5)
                                    val_scores_per_bag.append(inst_probs.cpu().numpy())
                                    val_true_counts.append(int(count.item()) if torch.is_tensor(count) else int(count))
                                    val_pred_pos_per_bag.append(bool(pred.cpu().item() == 1))
                            else:
                                Y_prob, predicted_label, _ = model(patches)

                                bag_label_f = bag_label.float()
                                Y_prob_clamped = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
                                loss = -1. * (bag_label_f * torch.log(Y_prob_clamped) + (1. - bag_label_f) * torch.log(1. - Y_prob_clamped))
                                val_loss += loss.item()
                                error = 1. - predicted_label.eq(bag_label_f).cpu().float().mean().data.item()
                                val_error += error

                                
                                y_pred.append(predicted_label.cpu().item())
                                y_prob.append(Y_prob.cpu().item())

                            y_true.append(bag_label.cpu().item())
                    val_loss /= len(val_dataset)
                    val_error /= len(val_dataset)   

                    metrics = calculate_metrics(y_true, y_pred, y_prob)
                    print('Epoch: {}, Val Loss: {:.4f}, Val Error: {:.4f}, Val AUC: {:.4f}\n'.format(
                        epoch, val_loss, val_error, metrics['auc']))    

                    # Log validation metrics to MLflow
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
                        "val_bias": metrics['bias']
                    }, step=epoch)      
                
                
                def test():
                    model.eval()
                    test_loss = 0.
                    test_error = 0.
                    y_true, y_pred, y_prob = [], [], []
                    count_truth, count_pred = [], []
                    attention_agg = []
                    all_instance_labels = []  
                    all_attention_weights = []  
                    all_thresholds = []

                    patch_inst_labels = []
                    patch_inst_scores = []

                    scores_per_bag = []   # Instanz-Scores je Test-Bag (Counting-Threshold-Eval)
                    true_counts = []      # wahre Counts je Test-Bag
                    pred_pos_per_bag = [] # Bag-Vorhersage je Test-Bag (fuer Gating)

                    with torch.no_grad():
                        for batch_idx, (patches, coords, label, count, instance_label) in enumerate(test_dataset):
                            if args.cuda:
                                patches, bag_label = patches.cuda(), label.cuda()
                            else:
                                bag_label = label
                            patches = patches.squeeze(0)  # Entfernt die Batch-Dimension, da sie 1 ist

                            if args.model == 'clam':
                                logits, Y_prob_full, predicted_label, A_raw, _ = model(patches)
                                prob_pos = Y_prob_full[0, 1]
                                loss = F.cross_entropy(logits, bag_label.long().view(-1))
                                test_loss += loss.item()
                                pred = predicted_label.view(-1).float()
                                error = 1. - pred.eq(bag_label.float().view(-1)).cpu().float().mean().item()
                                test_error += error
                                y_prob.append(prob_pos.cpu().item())
                                y_pred.append(pred.cpu().item())

                                # Instanz-Scores aus dem Instanz-Klassifikator (NICHT aus Attention)
                                _, inst_probs = model.count_positive_instances(patches.unsqueeze(0), threshold=0.5)

                                if args.count_threshold_eval:
                                    scores_per_bag.append(inst_probs.cpu().numpy())
                                    true_counts.append(int(count.item()) if torch.is_tensor(count) else int(count))
                                    pred_pos_per_bag.append(bool(pred.cpu().item() == 1))

                                inst_lbl = instance_label.cpu().numpy().flatten()
                                if len(set(inst_lbl.tolist())) > 2:        # MNIST Multi-Class -> binaer
                                    inst_lbl = (inst_lbl == 9).astype(int)
                                m = min(len(inst_lbl), inst_probs.shape[0])
                                patch_inst_labels.extend(inst_lbl[:m].tolist())
                                patch_inst_scores.extend(inst_probs[:m].cpu().numpy().tolist())
                            else:
                                # Single forward pass to avoid redindant computation
                                Y_prob, predicted_label, attention_weights = model(patches)
                                
                                bag_label_f = bag_label.float()
                                Y_prob_clamped = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
                                loss = -1. * (bag_label_f * torch.log(Y_prob_clamped) + (1. - bag_label_f) * torch.log(1. - Y_prob_clamped))
                                test_loss += loss.item()
                                error = 1. - predicted_label.eq(bag_label_f).cpu().float().mean().data.item()
                                test_error += error

                                y_pred.append(predicted_label.cpu().item())
                                y_prob.append(Y_prob.cpu().item())
                            
                            y_true.append(bag_label.cpu().item())

                            all_runs_results["seeds"].append(seed)
                            all_runs_results["bag_ids"].append(batch_idx)
                            all_runs_results["truth"].append(bag_label.cpu().item())
                            all_runs_results["predicted"].append(predicted_label.cpu().item())
                            
                            if args.log_attention_weights and args.model != 'clam':
                                attention_agg.append(attention_weights.cpu().numpy().tolist())
            
                            if args.naive_counting:
                                if predicted_label.cpu().item() == 1:  # Nur zählen, wenn die Vorhersage positiv ist
                                    if args.model == 'clam':
                                        predicted_count, _ = model.count_positive_instances(patches.unsqueeze(0))
                                        threshold = None
                                    else:
                                        predicted_count, _, threshold = model.count_positive_instances(patches)
                                else:
                                    predicted_count = 0
                                    threshold = 0
                                count_truth.append(count)
                                count_pred.append(predicted_count)
                                all_runs_results["count_threshold"].append(threshold)

                                if predicted_label.cpu().item() == 1 and args.model != 'clam':
                                    all_instance_labels.extend(instance_label.cpu().numpy().flatten().tolist()    )
                                    all_attention_weights.extend(attention_weights.cpu().numpy().flatten().tolist())
                                    all_thresholds.extend([threshold] * len(instance_label.cpu().numpy().flatten().tolist()))

                    test_loss /= len(test_dataset)
                    test_error /= len(test_dataset)

                    print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss, test_error))

                    metrics = calculate_metrics(y_true, y_pred, y_prob)
                    print('Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, '
                        'F1-Score: {:.4f}, AUC: {:.4f}'.format(
                            metrics['accuracy'], metrics['precision'], metrics['recall'],
                            metrics['f1_score'], metrics['auc']))

                    if (len(all_instance_labels) > 0 and len(all_attention_weights) > 0
                            and "count_threshold" in all_runs_results
                            and len(all_runs_results["count_threshold"]) > 0):
                        from sklearn.metrics import roc_auc_score, precision_score, recall_score
                        if len(set(all_instance_labels)) > 2:
                            all_instance_labels = np.where(np.array(all_instance_labels) == 9, 1, 0)  # Konvertiere Labels zu binär (positiv=1, negativ=0)
                      #      print("Weights:" + len(all_attention_weights).__str__() + "\n" + "Thresholds:" + len(all_runs_results["count_threshold"]).__str__())
                        all_attention_weights_converted = [1 if all_attention_weights[i] > all_thresholds[i] else 0 for i in range(len(all_attention_weights))]
                        conv_patch_auc = roc_auc_score(all_instance_labels, all_attention_weights_converted)

                        patch_auc = roc_auc_score(all_instance_labels, all_attention_weights)
                        patch_precision = precision_score(all_instance_labels, all_attention_weights_converted, zero_division=0)
                        patch_recall = recall_score(all_instance_labels, all_attention_weights_converted, zero_division=0)
                        mlflow.log_metrics({
                            'patch_level_auc': patch_auc,
                            'patch_level_precision': patch_precision,
                            'patch_level_recall': patch_recall,
                            'patch_level_auc_converted': conv_patch_auc,
                        })
                        metrics['patch_level_auc'] = patch_auc
                        metrics['patch_level_precision'] = patch_precision
                        metrics['patch_level_recall'] = patch_recall

                        metrics['patch_level_auc_converted'] = conv_patch_auc


                        print(f'Patch-Level AUC (conv): {patch_auc:.4f} ({conv_patch_auc:.4f}), Precision: {patch_precision:.4f} , Recall: {patch_recall:.4f}')

                    elif args.model == 'clam' and len(patch_inst_labels) > 0:
                        from sklearn.metrics import roc_auc_score, precision_score, recall_score
                        inst_labels_arr = np.array(patch_inst_labels)
                        inst_scores_arr = np.array(patch_inst_scores)
                        inst_preds = (inst_scores_arr >= 0.5).astype(int)
                        if len(set(patch_inst_labels)) > 1:
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
                        metrics['patch_level_auc'] = patch_auc
                        metrics['patch_level_precision'] = patch_precision
                        metrics['patch_level_recall'] = patch_recall
                        print(f'Patch-Level AUC (inst. clf.): {patch_auc:.4f}, '
                              f'Precision: {patch_precision:.4f}, Recall: {patch_recall:.4f}')
                    else:
                        patch_auc = 0
                        patch_precision = 0
                        patch_recall = 0
                        print('Patch-Level AUC/Precision/Recall: Could not be computed (no positive bags)')

                    mlflow.log_metrics({"test_loss": test_loss,
                                        "test_error": test_error,
                                        "accuracy": metrics['accuracy'],
                                        "precision": metrics['precision'],
                                        "recall": metrics['recall'],
                                        "f1_score": metrics['f1_score'],
                                        "auc": metrics['auc'],
                                        "mae": metrics['mae'],
                                        "rmse": metrics['rmse'],
                                        "bias": metrics['bias']})
                    
                    metrics['test_loss'] = test_loss
                    metrics['test_error'] = test_error
                    metrics['patch_level_auc'] = patch_auc
                    metrics['patch_level_precision'] = patch_precision
                    metrics['patch_level_recall'] = patch_recall

                    if args.naive_counting and len(count_pred) > 0:
                        counting_metrics = calculate_counting_metrics(count_truth, count_pred)
                        metrics['counting_accuracy'] = counting_metrics['counting_accuracy']
                        metrics['counting_mae'] = counting_metrics['counting_mae']
                        metrics['counting_rmse'] = counting_metrics['counting_rmse']
                        mlflow.log_metric('counting_accuracy', counting_metrics['counting_accuracy'])

                        clean_truth = [int(c[0].item()) if isinstance(c, list) and torch.is_tensor(c[0]) 
                                                        else int(c.item()) if torch.is_tensor(c) else int(c) for c in count_truth]

                        clean_pred = [int(p.item()) if torch.is_tensor(p) else int(p) for p in count_pred]

                        # table_data =  {"Truth": clean_truth, "Predicted": clean_pred}
                        # mlflow.log_table(table_data, artifact_file="counting_results.json")

                        all_runs_results["count_truth"].extend(clean_truth)
                        all_runs_results["count_pred"].extend(clean_pred)

                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.scatter(clean_truth, clean_pred, alpha=0.6, edgecolors='w')

                        max_val = max(max(clean_truth), max(clean_pred))
                        ax.plot([0, max_val], [0, max_val], 'r--') 

                        ax.set_xlabel("True Count")
                        ax.set_ylabel("Predicted Count")
                        ax.set_title("Count Truth vs. Prediction")

                        mlflow.log_figure(fig, "counting_plot.png")
                        plt.close(fig) 

                        mlflow.log_metrics({"count_mae": counting_metrics['counting_mae'], "count_rmse": counting_metrics['counting_rmse']})
                        print('Counting Accuracy: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}'.format(
                            counting_metrics['counting_accuracy'], counting_metrics['counting_mae'], counting_metrics['counting_rmse']))
                        
                    # ── Counting-Threshold-Auswertung (nur CLAM) ──────────────────────
                    # Alle Threshold-Strategien sind bag-gegatet: negativ klassifizierte
                    # Bags zaehlen 0. Das macht den Vergleich fair zum naiven (ebenfalls
                    # gegateten) Zaehlen und passt zur zero-inflated Count-Verteilung.
                    if args.count_threshold_eval and args.model == 'clam' and len(scores_per_bag) > 0:
                        gate = np.asarray(pred_pos_per_bag, dtype=bool)

                        # 1) Globaler Sweep -- Bias/MAE-Kurve ueber Thresholds (gegatet)
                        print("\n--- Counting-Threshold-Sweep (Test, bag-gegatet) ---")
                        for thr in np.arange(0.20, 0.75, 0.05):
                            bias, mae = CLAM.counting_scores_per_bag(scores_per_bag, true_counts, thr, pred_pos=gate)
                            print(f"  thr={thr:.2f}  Bias={bias:+.2f}  MAE={mae:.2f}")
                            mlflow.log_metric("count_sweep_bias", bias, step=int(thr * 100))
                            mlflow.log_metric("count_sweep_mae",  mae,  step=int(thr * 100))

                        # 2) Otsu pro Bag (gegatet)
                        otsu_bias, otsu_mae = CLAM.counting_scores_otsu(scores_per_bag, true_counts, pred_pos=gate)
                        print(f"  Otsu (per-bag)  Bias={otsu_bias:+.2f}  MAE={otsu_mae:.2f}")
                        mlflow.log_metric("count_otsu_bias", otsu_bias)
                        mlflow.log_metric("count_otsu_mae",  otsu_mae)
                        metrics['count_otsu_mae'] = otsu_mae

                        # 3) Auf Validierung bias-kalibrierter globaler Threshold (Val + Test gegatet)
                        if len(val_scores_per_bag) > 0:
                            val_gate = np.asarray(val_pred_pos_per_bag, dtype=bool)
                            cal_thr, cal_bias_val, cal_mae_val = CLAM.calibrate_threshold(
                                val_scores_per_bag, val_true_counts, pred_pos=val_gate)
                            test_bias, test_mae = CLAM.counting_scores_per_bag(scores_per_bag, true_counts, cal_thr, pred_pos=gate)
                            print(f"  Kalibriert (thr={cal_thr:.2f} aus Val)  Test-Bias={test_bias:+.2f}  Test-MAE={test_mae:.2f}")
                            mlflow.log_metric("count_calibrated_threshold", cal_thr)
                            mlflow.log_metric("count_calibrated_bias", test_bias)
                            mlflow.log_metric("count_calibrated_mae",  test_mae)
                            metrics['count_calibrated_threshold'] = cal_thr
                            metrics['count_calibrated_mae'] = test_mae
                        else:
                            print("  Kalibrierung uebersprungen (keine Val-Scores gesammelt).")

                        # 4) Soft-Count: Summe der Instanz-Wahrscheinlichkeiten (gegatet, threshold-frei)
                        if args.soft_counting:
                            soft_bias, soft_mae = CLAM.counting_scores_soft(scores_per_bag, true_counts, pred_pos=gate)
                            print(f"  Soft-Count (Sum P, gegatet)  Bias={soft_bias:+.2f}  MAE={soft_mae:.2f}")
                            mlflow.log_metric("count_soft_bias", soft_bias)
                            mlflow.log_metric("count_soft_mae",  soft_mae)
                            metrics['count_soft_mae'] = soft_mae

                        # 5) Trivialer Baseline-Schaetzer: fester Anteil p_train * N (ungegatet, Trivial-Anker)
                        if p_train is not None:
                            pred = np.array([p_train * len(s) for s in scores_per_bag])
                            true = np.array(true_counts)
                            base_bias = float((pred - true).mean())
                            base_mae  = float(np.abs(pred - true).mean())
                            print(f"  Baseline (p={p_train:.2f} * N)  Bias={base_bias:+.2f}  MAE={base_mae:.2f}")
                            mlflow.log_metric("count_baseline_bias", base_bias)
                            mlflow.log_metric("count_baseline_mae", base_mae)
                            metrics['count_baseline_mae'] = base_mae
                    # ──────────────────────────────────────────────────────────────────

                    if args.log_attention_weights and attention_agg:
                        all_runs_results["attention_weights"].extend(attention_agg)

                    return metrics, count_truth, count_pred
                
                print('Starting training!')
                for epoch in range(1, args.epochs + 1):
                    train(epoch)
                    validate(epoch)
                    
                print('Starting testing!')
                metrics, seed_truth, seed_pred = test()

                all_metrics.append(metrics)
                mlflow.pytorch.log_model(model, "model")  # Log the model to MLflow

                # ── Feature-Visualisierung (Child Run / Per-Seed) ────────────────────
                if args.visualize_features:
                    if not hasattr(model, 'extract_features'):
                        print("Warning: Modell hat keine extract_features()-Methode. "
                              "Bitte model.py aktualisieren. Visualisierung übersprungen.")
                    else:
                        try:
                            print(f"\nSammle Features für Visualisierung...")
                            H, A, bag_lbls, inst_lbls, bag_ids = vf.collect_features_from_loader(
                                model, test_dataset,
                                device="cuda" if args.cuda else "cpu"
                            )
                            H_2d = vf.reduce_dimensions(H, "umap")
                            fig = vf.plot_per_seed(
                                H_2d, A, bag_lbls, inst_lbls,
                                title_suffix=f"Seed {seed}"
                            )
                            mlflow.log_figure(fig, "feature_visualization.png")
                            plt.close(fig)
                            print(f"  Feature-Plot in Child Run geloggt (seed={seed}).")
 
                            # Rohdaten für den aggregierten Parent-Plot aufbewahren
                            all_feature_data.append({
                                "H":               H,
                                "A":               A,
                                "bag_labels":      bag_lbls,
                                "instance_labels": inst_lbls,
                                "seed_ids":        np.full(len(H), seed, dtype=int),
                            })
                        except Exception as viz_err:
                            print(f"Feature-Visualisierung für Seed {seed} fehlgeschlagen: {viz_err}")
                # ────────────────────────────────────────────────────────────────────

        # Logging in parent run after all seeds have been processed
        mlflow.log_table(all_runs_results, artifact_file="aggregated_run_results.json")
        mlflow.set_tags(model_tags)

        if all_metrics:
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
            
        if all_runs_results.get("count_truth") and all_runs_results.get("count_pred"):
            print("Logging aggregated counting artifacts...")
            
            fig, ax = plt.subplots(figsize=(8, 8))
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'brown']
            
            unique_seeds = sorted(set(all_runs_results["seeds"]))
            for i, seed in enumerate(unique_seeds):
                seed_truth = [all_runs_results["count_truth"][i] for i, s in enumerate(all_runs_results["seeds"]) if s == seed]
                seed_pred = [all_runs_results["count_pred"][i] for i, s in enumerate(all_runs_results["seeds"]) if s == seed]
                
                # Nimm die entsprechende Farbe (modulo-Operator falls mehr Seeds als Farben existieren)
                color = colors[i % len(colors)]
                
                ax.scatter(
                    seed_truth, seed_pred, 
                    color=color, 
                    label=f"Seed {seed}",
                    alpha=0.6, edgecolors='w'
                )
            
            max_val = max(max(all_runs_results["count_truth"]), max(all_runs_results["count_pred"]))
            ax.plot([0, max_val], [0, max_val], 'r--', label="Perfect Prediction") 
            
            ax.set_xlabel("True Count")
            ax.set_ylabel("Predicted Count")
            ax.set_title("Aggregated Count: Truth vs. Prediction (All Seeds)")
            ax.legend()
            
            mlflow.log_figure(fig, "aggregated_counting_plot.png")
            plt.close(fig)

        # ── Aggregierte Feature-Visualisierung (Parent Run) ──────────────────────
        if args.visualize_features and len(all_feature_data) > 0:
            try:
                print("\nErstelle aggregierten Feature-Plot über alle Seeds...")
                H_agg   = np.concatenate([d["H"]               for d in all_feature_data])
                A_agg   = np.concatenate([d["A"]               for d in all_feature_data])
                bl_agg  = np.concatenate([d["bag_labels"]      for d in all_feature_data])
                il_agg  = np.concatenate([d["instance_labels"] for d in all_feature_data])
                sid_agg = np.concatenate([d["seed_ids"]        for d in all_feature_data])
 
                print(f"  Gesamt: {len(H_agg)} Instanzen aus {len(all_feature_data)} Seed(s).")
                H_2d_agg = vf.reduce_dimensions(H_agg, "umap")
 
                if len(all_feature_data) > 1:
                    # Mehrere Seeds → 4-Panel-Plot mit Seed-Zugehörigkeit
                    fig = vf.plot_aggregated(H_2d_agg, A_agg, bl_agg, il_agg, sid_agg)
                    mlflow.log_figure(fig, "feature_visualization_aggregated.png")
                else:
                    # Nur ein Seed → gleicher 3-Panel-Plot wie im Child Run,
                    # aber explizit im Parent Run geloggt
                    fig = vf.plot_per_seed(
                        H_2d_agg, A_agg, bl_agg, il_agg,
                        title_suffix=f"Seed {int(sid_agg[0])} (Parent)"
                    )
                    mlflow.log_figure(fig, "feature_visualization_aggregated.png")
 
                plt.close(fig)
                print("  Aggregierter Feature-Plot in Parent Run geloggt.")
            except Exception as viz_err:
                print(f"Aggregierte Feature-Visualisierung fehlgeschlagen: {viz_err}")
        # ────────────────────────────────────────────────────────────────────────

    except Exception as e:
        print(f"An error occurred: {e}")
        mlflow.log_param('error_message', str(e))
        mlflow.set_tags({"status": "failed"})
    else:
        mlflow.set_tags({"status": "completed"})


    