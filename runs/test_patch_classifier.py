import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch, torch.nn as nn
import torch.utils.data as data_utils
from sklearn.metrics import roc_auc_score
from data.data_management.dataset_manager import DatasetReader

device = "cuda" if torch.cuda.is_available() else "cpu"
PATH = "../data/datasets/bags/gwhd_bags_dense.h5"
DATASET = "gwhd_bags_dense"
# Gleicher Backbone wie im MIL-Modell – fairer Kapazitätsvergleich
class PatchClassifier(nn.Module):
    def __init__(self, num_maps=50, kernel_size=5, pool_size=4, M=500, in_ch=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 20, kernel_size, padding=kernel_size//2), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, num_maps, kernel_size, padding=kernel_size//2), nn.ReLU(),
            nn.AdaptiveMaxPool2d((pool_size, pool_size)),
        )
        self.head = nn.Sequential(
            nn.Linear(num_maps*pool_size*pool_size, M), nn.ReLU(),
            nn.Linear(M, 1),
        )
    def forward(self, x):
        h = self.features(x).flatten(1)
        return self.head(h).squeeze(-1)

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

train_loader = data_utils.DataLoader(DatasetReader(PATH, dataset_name=DATASET, split='train'), batch_size=1)
test_loader  = data_utils.DataLoader(DatasetReader(PATH, dataset_name=DATASET, split='test'),  batch_size=1)

Xtr, ytr = flatten_patches(train_loader)
Xtr, ytr = Xtr[:10000], ytr[:10000]  # Beschränkung auf 10k Patches für schnellere Tests
Xte, yte = flatten_patches(test_loader)
Xte, yte = Xte[:2000], yte[:2000]  # Beschränkung auf 2k Patches für schnellere Tests
print(f"Train-Patches: {len(ytr)} ({ytr.mean():.1%} positiv) | Test: {len(yte)}")

model = PatchClassifier(in_ch=Xtr.shape[1]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.BCEWithLogitsLoss()
ds = data_utils.TensorDataset(Xtr, ytr)
dl = data_utils.DataLoader(ds, batch_size=128, shuffle=True)

for epoch in range(15):
    model.train()
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()

model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(Xte.to(device))).cpu().numpy()
print(f"Supervised Patch-AUC (obere Schranke): {roc_auc_score(yte.numpy(), probs):.4f}")