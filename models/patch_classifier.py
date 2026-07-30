import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.utils.data as data_utils

from sklearn.metrics import roc_auc_score

from data.data_management.dataset_manager import DatasetReader
from models.backbone import Backbone 
from models.learned_grayscale import LearnedGrayscale

class PatchClassifier(nn.Module):
    def __init__(self, in_channels=3, kernel_size=3, num_maps=64, pool_size=4, M=128, grayscaling=False):
        super().__init__()
        self.backbone = Backbone(in_channels=in_channels, 
            kernel_size=kernel_size, 
            num_maps=num_maps, 
            pool_size=pool_size, 
            M=M, 
            grayscaling=grayscaling
        )

        self.head = nn.Sequential(
            nn.Linear(M, 1),
        )

    def forward(self, x):
        H = self.backbone(x)  # KxM
        return self.head(H).squeeze(-1)  # K

def flatten_patches(loader, target_digit=None):
    """Zieht alle Patches + Instanz-Labels aus dem Bag-Loader."""
    Xs, ys = [], []
    for patches, coords, label, count, inst in loader:
        patches = patches.squeeze(0)                 # [K, C, H, W]
        inst = inst.cpu().flatten().numpy()
        if target_digit is not None:                 # MNIST: Multi-Class -> binär
            inst = (inst == target_digit).astype(int)
        Xs.append(patches.cpu()); ys.append(torch.tensor(inst[:patches.shape[0]]))
    return torch.cat(Xs), torch.cat(ys).float()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Backbone Pre-training on Patches')

    parser.add_argument('--path', type=str, default='../data/datasets/bags/gwhd_bags_dense.h5', help='Path to dataset')
    parser.add_argument('--dataset', type=str, default='gwhd_bags_dense', help='Dataset name')
    parser.add_argument('--output_dir', type=str, default='./state_dicts', help='Directory to save the trained model')
    parser.add_argument('--grayscaling', action='store_true', help='Use learned grayscaling', default=False)
    parser.add_argument('--kernel_size', type=int, default=5, help='Size of the convolutional kernel')
    parser.add_argument('--num_maps', type=int, default=64, help='Number of feature maps in the backbone')
    parser.add_argument('--pool_size', type=int, default=4, help='Size of the adaptive pooling layer')
    parser.add_argument('--M', type=int, default=500, help='Size of the fully connected layer in the backbone')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--no_cuda', action='store_true', help='Don\'t use CUDA', default=False)

    args = parser.parse_args()

    train_loader = data_utils.DataLoader(DatasetReader(args.path, dataset_name=args.dataset, split='train'), batch_size=1)
    test_loader  = data_utils.DataLoader(DatasetReader(args.path, dataset_name=args.dataset, split='test'),  batch_size=1)

    Xtr, ytr = flatten_patches(train_loader)
    Xtr, ytr = Xtr[:10000], ytr[:10000]  # Beschränkung auf 10k Patches für schnellere Tests
    Xte, yte = flatten_patches(test_loader)
    Xte, yte = Xte[:2000], yte[:2000]  # Beschränkung auf 2k Patches für schnellere Tests
    print(f"Train-Patches: {len(ytr)} ({ytr.mean():.1%} positiv) | Test: {len(yte)}")

    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    model = PatchClassifier(
        in_channels=Xtr.shape[1],
        kernel_size=args.kernel_size,
        num_maps=args.num_maps,
        pool_size=args.pool_size,
        M=args.M,
        grayscaling=args.grayscaling
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    ds = data_utils.TensorDataset(Xtr, ytr)
    dl = data_utils.DataLoader(ds, batch_size=128, shuffle=True)

    for epoch in range(args.epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            y_pred = model(Xte.to(device)).cpu()
            auc = roc_auc_score(yte.numpy(), y_pred.numpy())
            print(f"Epoch {epoch+1}/{args.epochs} | Test AUC: {auc:.4f}")

    torch.save(model.backbone.state_dict(), f"{args.output_dir}/patch_classifier_{args.dataset}.pth")
    print(f"Model saved to {args.output_dir}/patch_classifier_{args.dataset}.pth")