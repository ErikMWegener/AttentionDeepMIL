import torch
import torch.nn as nn
import torch.nn.functional as F
from models import otsu


class Attention(nn.Module):
    def __init__(self, M=500, L=128, num_maps=50, kernel_size=5, pool_size=4, ATTENTION_BRANCHES=1, in_channels=1, attention_activation="softmax"):
        super(Attention, self).__init__()
        self.M = M
        self.L = L
        self.num_maps = num_maps
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.ATTENTION_BRANCHES = ATTENTION_BRANCHES
        self.attention_activation = attention_activation

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(in_channels, 20, kernel_size=self.kernel_size, padding=self.kernel_size//2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, self.num_maps, kernel_size=self.kernel_size, padding=self.kernel_size//2),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((self.pool_size, self.pool_size))
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(self.num_maps * self.pool_size * self.pool_size, self.M),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

        if self.attention_activation == "sigmoid" or self.attention_activation == "min_max":
            self.z_norm = nn.LayerNorm(self.M)
        

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, self.num_maps * self.pool_size * self.pool_size)
        H = self.feature_extractor_part2(H)  # KxM

        A = self.attention(H)  # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        if self.attention_activation == "sigmoid":
            A = (A - A.mean()) / (A.std() + 1e-8)
            A = F.sigmoid(A)  # sigmoid over K
        elif self.attention_activation == "min_max":
            A_min = A.min(dim=1, keepdim=True)[0]
            A_max = A.max(dim=1, keepdim=True)[0]
            A = (A - A_min) / (A_max - A_min + 1e-8)  # min-max normalize to [0, 1]
        else:
            A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        if self.attention_activation == "sigmoid" or self.attention_activation == "min_max":
            avg_attention = torch.mean(A, dim=1, keepdim=True)  # Durchschnitt über die K-Instanzen für jede ATTENTION_BRANCHES
            scale = torch.clamp(avg_attention, min=0.05)  # Verhindert Division durch Null
            Z = Z / scale  # Skaliere Z entsprechend der durchschnittlichen Aufmerksamkeit

            Z = self.z_norm(Z)  # Normalisiere Z, um Stabilität zu gewährleisten

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

    def count_positive_instances(self, X, threshold=None):
        """Count instances in a bag whose attention weight exceeds a threshold.

        Args:
            X: Input bag tensor.
            threshold: Attention weight threshold. If None, uses 1/K
                (uniform attention), where K is the number of instances.

        Returns:
            count: Number of instances with attention weight above the threshold.
            attention_weights: The raw attention weights for all instances.
        """
        _, _, A = self.forward(X)
        K = A.shape[1]  # number of instances
        if threshold is None:
            if self.attention_activation == "sigmoid":
                threshold = torch.mean(A).item()
            elif self.attention_activation == "min_max":
                threshold = otsu.compute_otsu_threshold(A)
            else:
                threshold = 1.0 / K
        count = int((A.squeeze(0) > threshold).sum().item())
        return count, A, threshold

class GatedAttention(nn.Module):
    def __init__(self, M=500, L=128, num_maps=50, kernel_size=5, pool_size=4, ATTENTION_BRANCHES=1, in_channels=1, attention_activation="softmax"):
        super(GatedAttention, self).__init__()
        self.M = M
        self.L = L
        self.num_maps = num_maps
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.ATTENTION_BRANCHES = ATTENTION_BRANCHES
        self.attention_activation = attention_activation

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(in_channels, 20, kernel_size=self.kernel_size, padding=self.kernel_size//2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, self.num_maps, kernel_size=self.kernel_size, padding=self.kernel_size//2),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((self.pool_size, self.pool_size))
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(self.num_maps * self.pool_size * self.pool_size, self.M),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

        if self.attention_activation == "sigmoid":
            self.z_norm = nn.LayerNorm(self.M)

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, self.num_maps * self.pool_size * self.pool_size)
        H = self.feature_extractor_part2(H)  # KxM

        A_V = self.attention_V(H)  # KxL
        A_U = self.attention_U(H)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        if self.attention_activation == "sigmoid":
            A = F.sigmoid(A)  # sigmoid over K
        else:
            A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        if self.attention_activation == "sigmoid":
            avg_attention = torch.mean(A, dim=1, keepdim=True)
            scale = torch.clamp(avg_attention, min=0.05)
            Z = Z / scale
            Z = self.z_norm(Z)

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

    def count_positive_instances(self, X, threshold=None):
        """Count instances in a bag whose attention weight exceeds a threshold.

        Args:
            X: Input bag tensor.
            threshold: Attention weight threshold. If None, uses 1/K
                (uniform attention), where K is the number of instances.

        Returns:
            count: Number of instances with attention weight above the threshold.
            attention_weights: The raw attention weights for all instances.
        """
        _, _, A = self.forward(X)
        K = A.shape[1]  # number of instances
        if threshold is None:
            if self.attention_activation == "sigmoid":
                threshold = torch.mean(A).item()
            else:
                threshold = 1.0 / K
        count = int((A.squeeze(0) > threshold).sum().item())
        return count, A, threshold