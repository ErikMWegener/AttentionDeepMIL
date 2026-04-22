import sys
import os

# Fügt das Stammverzeichnis des Projekts zum Python-Pfad hinzu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import torch
import torch.utils.data as data_utils

from data.data_management.dataset_manager import DatasetReader
from eval.scripts.metrics import calculate_metrics, save_results_to_csv
from models.model import Attention, GatedAttention

# Get arguments from command line
parser = argparse.ArgumentParser(description='Testing Atteintion MIL models on datasets loaded from H5 files.')

parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--naive_counting', action='store_true', default=False,
                    help='activates counting of positve instances for model testing')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='attention', 
                    help='Choose b/w attention and gated_attention')
parser.add_argument('--results_csv', type=str, default='results.csv', metavar='CSV',
                    help='path to CSV file for logging results (default: results.csv)')
parser.add_argument('--dataset', type=str, default='mnist_bags.h5', metavar='H5', 
                    help='path to H5 file containing the dataset (default: mnist_bags.h5)')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

# Load dataset using DatasetReader
print('Load Train and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_dataset = data_utils.DataLoader(DatasetReader(args.dataset, dataset_name='mnist_bags', split='train'),
                                      batch_size=1, 
                                      shuffle=True,
                                      **loader_kwargs)
test_dataset = data_utils.DataLoader(DatasetReader(args.dataset, dataset_name='mnist_bags', split='test'),
                                     batch_size=1,
                                     shuffle=False,
                                     **loader_kwargs)

print('Initialize model')
if args.model == 'attention':
    model = Attention()
elif args.model == 'gated_attention':
    model = GatedAttention()
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

    print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss, train_error))


def test():
    model.eval()
    test_loss = 0.
    test_error = 0.
    y_true, y_pred, y_prob = [], [], []
    counting_correct, counting_total = 0, 0

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
                if predicted_label.cpu().item() == bag_label.cpu().item():
                    predicted_count, _ = model.count_positive_instances(patches)
                else:
                    predicted_count = 0
                true_count = count
                if predicted_count == true_count:
                    counting_correct += 1
                counting_total += 1

    test_loss /= len(test_dataset)
    test_error /= len(test_dataset)

    print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss, test_error))

    metrics = calculate_metrics(y_true, y_pred, y_prob)
    print('Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, '
          'F1-Score: {:.4f}, AUC: {:.4f}'.format(
              metrics['accuracy'], metrics['precision'], metrics['recall'],
              metrics['f1_score'], metrics['auc']))

    if args.naive_counting and counting_total > 0:
        counting_accuracy = counting_correct / counting_total
        metrics['counting_accuracy'] = counting_accuracy
        print('Counting Accuracy: {:.4f}'.format(counting_accuracy))

    config = {
        'dataset': args.dataset,
        'model': args.model,
        'epochs': args.epochs,
        'lr': args.lr,
        'reg': args.reg,
        'seed': args.seed,
        'test_loss': test_loss,
        'test_error': test_error,
    }
    save_results_to_csv(args.results_csv, config, metrics)


if __name__ == "__main__":
    print('Start Training')
    for epoch in range(1, args.epochs + 1):
        train(epoch)
    print('Start Testing')
    test()

    