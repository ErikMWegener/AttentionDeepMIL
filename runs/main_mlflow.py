import sys
import os

# Fügt das Stammverzeichnis des Projekts zum Python-Pfad hinzu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import torch
import torch.utils.data as data_utils
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import yaml


from data.data_management.dataset_manager import DatasetReader
from eval.scripts.metrics import calculate_metrics, calculate_counting_metrics
from models.model import Attention, GatedAttention

# Get arguments from command line
parser = argparse.ArgumentParser(description='Testing Atteintion MIL models on datasets loaded from H5 files.')

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
parser.add_argument('--model', type=str, default='attention', 
                    help='Choose b/w attention and gated_attention')
parser.add_argument('--dataset', type=str, default='mnist_bags', metavar='H5', 
                    help='path to H5 file containing the dataset (default: mnist_bags.h5)')
parser.add_argument('--path', type=str, default='../data/datasets/bags/mnist_bags.h5', metavar='H5',
                    help='path to H5 file containing the dataset (default: mnist_bags.h5)')
parser.add_argument('--exp_name', type=str, default=None, metavar='EXP',
                    help='name of the MLflow experiment (default: default)')
parser.add_argument('--sigmoid_attention', action='store_true', default=False,
                    help='use sigmoid instead of softmax for attention weights')
parser.add_argument('--rgb', action='store_true', default=False,
                    help='use RGB input instead of grayscale')

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
    config_filename = f"configs/{args.model}_{args.dataset}_lr{args.lr}_reg{args.reg}_ep{args.epochs}_sigmoid{args.sigmoid_attention}_config.yaml"
    with open(config_filename, 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False)
    print(f"Run configuration saved to {config_filename}")

if args.exp_name:
    mlflow.set_experiment(args.exp_name)
# else:
#     experiment_name = f"{args.model}_{args.dataset}"
#     mlflow.set_experiment(experiment_name)

# Starting partent MLflow run to log parameters and artifacts common to all seeds
with mlflow.start_run(run_name=f"{args.model}_{args.dataset}_lr{args.lr}_reg{args.reg}_ep{args.epochs}_sigmoid{args.sigmoid_attention}") as parent_run:
    mlflow.log_params(vars(args))
    if config_args.config is not None:
        mlflow.log_artifact(config_args.config)
    if config_args.config is None:
        mlflow.log_artifact(config_filename)

    mlflow.set_tags({
        "model": args.model,
        "dataset": args.dataset,
        "num_seeds": len(args.seeds),
        #"hyperparams": "baseline",
        #"comparison_group": "v1.0",
        "status": "in_progress"
    })

    all_metrics = []

    all_count_truth = []
    all_count_pred = []
    all_count_seed = []
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
                    model = Attention(in_channels=3 if args.rgb else 1, sigmoid_attention=args.sigmoid_attention)
                elif args.model == 'gated_attention':
                    model = GatedAttention(in_channels=3 if args.rgb else 1, sigmoid_attention=args.sigmoid_attention)
                if args.cuda:
                    model.cuda()

                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)


                def train(epoch):
                    model.train()
                    train_loss = 0.
                    train_error = 0.
                    for batch_idx, (patches, coords, label, count, instance_label) in enumerate(train_dataset):
                        bag_label = label
                        if args.cuda:
                            patches, bag_label = patches.cuda(), bag_label.cuda()

                        patches = patches.squeeze(0)  # Entfernt die Batch-Dimension, da sie 1 ist

                        # reset gradients
                        optimizer.zero_grad()
                        # calculate loss and metrics
                        loss, _ = model.calculate_objective(patches, bag_label)
                        train_loss += loss.item()
                        error, _ = model.calculate_classification_error(patches, bag_label)
                        train_error += error
                        # backward pass
                        loss.backward()
                        # step
                        optimizer.step()

                    # calculate loss and error for epoch
                    train_loss /= len(train_dataset)
                    train_error /= len(train_dataset)

                    # Log training loss and error to MLflow
                    mlflow.log_metric('train_loss', train_loss, step=epoch)
                    mlflow.log_metric('train_error', train_error, step=epoch)

                    print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss, train_error))

                def validate(epoch):
                    model.eval()
                    val_loss = 0.
                    val_error = 0.
                    y_true, y_pred, y_prob = [], [], []

                    with torch.no_grad():
                        for batch_idx, (patches, coords, label, count, instance_label) in enumerate(val_dataset):
                            if args.cuda:
                                patches, bag_label = patches.cuda(), label.cuda()
                            else: 
                                bag_label = label
                            
                            patches = patches.squeeze(0)

                            Y_prob, predicted_label, _ = model(patches)

                            bag_label_f = bag_label.float()
                            Y_prob_clamped = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
                            loss = -1. * (bag_label_f * torch.log(Y_prob_clamped) + (1. - bag_label_f) * torch.log(1. - Y_prob_clamped))
                            val_loss += loss.item()
                            error = 1. - predicted_label.eq(bag_label_f).cpu().float().mean().data.item()
                            val_error += error

                            y_true.append(bag_label.cpu().item())
                            y_pred.append(predicted_label.cpu().item())
                            y_prob.append(Y_prob.cpu().item())

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

                    with torch.no_grad():
                        for batch_idx, (patches, coords, label, count, instance_label) in enumerate(test_dataset):
                            if args.cuda:
                                patches, bag_label = patches.cuda(), label.cuda()

                            patches = patches.squeeze(0)  # Entfernt die Batch-Dimension, da sie 1 ist

                            # Single forward pass to avoid redindant computation
                            Y_prob, predicted_label, attention_weights = model(patches)

                            bag_label_f = bag_label.float()
                            Y_prob_clamped = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
                            loss = -1. * (bag_label_f * torch.log(Y_prob_clamped) + (1. - bag_label_f) * torch.log(1. - Y_prob_clamped))
                            test_loss += loss.item()
                            error = 1. - predicted_label.eq(bag_label_f).cpu().float().mean().data.item()
                            test_error += error

                            y_true.append(bag_label.cpu().item())
                            y_pred.append(predicted_label.cpu().item())
                            y_prob.append(Y_prob.cpu().item())

                            if args.naive_counting:
                                if predicted_label.cpu().item() == 1:  # Nur zählen, wenn die Vorhersage positiv ist
                                    predicted_count, _ = model.count_positive_instances(patches)
                                else:
                                    predicted_count = 0
                                count_truth.append(count)
                                count_pred.append(predicted_count)

                    test_loss /= len(test_dataset)
                    test_error /= len(test_dataset)

                    print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss, test_error))

                    metrics = calculate_metrics(y_true, y_pred, y_prob)
                    print('Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, '
                        'F1-Score: {:.4f}, AUC: {:.4f}'.format(
                            metrics['accuracy'], metrics['precision'], metrics['recall'],
                            metrics['f1_score'], metrics['auc']))

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

                    if args.naive_counting and len(count_pred) > 0:
                        counting_metrics = calculate_counting_metrics(count_truth, count_pred)
                        metrics['counting_accuracy'] = counting_metrics['counting_accuracy']
                        metrics['counting_mae'] = counting_metrics['counting_mae']
                        metrics['counting_rmse'] = counting_metrics['counting_rmse']
                        mlflow.log_metric('counting_accuracy', counting_metrics['counting_accuracy'])

                        clean_truth = [int(c[0].item()) if isinstance(c, list) and torch.is_tensor(c[0]) 
                                                        else int(c.item()) if torch.is_tensor(c) else int(c) for c in count_truth]

                        clean_pred = [int(p.item()) if torch.is_tensor(p) else int(p) for p in count_pred]

                        table_data =  {"Truth": clean_truth, "Predicted": clean_pred}
                        mlflow.log_table(table_data, artifact_file="counting_results.json")

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
                        return metrics, clean_truth, clean_pred
                    
                    return metrics, [], []
                
                print('Starting training!')
                for epoch in range(1, args.epochs + 1):
                    train(epoch)
                    validate(epoch)
                    
                print('Starting testing!')
                metrics, seed_truth, seed_pred = test()
                
                if seed_truth and seed_pred:
                    all_count_truth.extend(seed_truth)
                    all_count_pred.extend(seed_pred)
                    all_count_seed.extend([seed] * len(seed_truth))

                all_metrics.append(metrics)
                mlflow.pytorch.log_model(model, "model")  # Log the model to MLflow


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
            
            if all_count_truth and all_count_pred:
                print("Logging aggregated counting artifacts...")
                
                parent_table = {
                    "Seed": all_count_seed,
                    "Truth": all_count_truth,
                    "Predicted": all_count_pred
                }
                mlflow.log_table(parent_table, artifact_file="aggregated_counting_results.json")
                
                fig, ax = plt.subplots(figsize=(8, 8))

                colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'brown']
                
                unique_seeds = sorted(set(all_count_seed))
                for i, seed in enumerate(unique_seeds):
                    seed_truth = [all_count_truth[i] for i, s in enumerate(all_count_seed) if s == seed]
                    seed_pred = [all_count_pred[i] for i, s in enumerate(all_count_seed) if s == seed]
                    
                    # Nimm die entsprechende Farbe (modulo-Operator falls mehr Seeds als Farben existieren)
                    color = colors[i % len(colors)]
                    
                    ax.scatter(
                        seed_truth, seed_pred, 
                        color=color, 
                        label=f"Seed {seed}",
                        alpha=0.6, edgecolors='w'
                    )
                
                max_val = max(max(all_count_truth), max(all_count_pred))
                ax.plot([0, max_val], [0, max_val], 'r--', label="Perfect Prediction") 
                
                ax.set_xlabel("True Count")
                ax.set_ylabel("Predicted Count")
                ax.set_title("Aggregated Count: Truth vs. Prediction (All Seeds)")
                ax.legend()
                
                mlflow.log_figure(fig, "aggregated_counting_plot.png")
                plt.close(fig)

    except Exception as e:
        print(f"An error occurred: {e}")
        mlflow.log_param('error_message', str(e))
        mlflow.set_tags({"status": "failed"})
    else:
        mlflow.set_tags({"status": "completed"})


    