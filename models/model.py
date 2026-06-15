import torch
import torch.nn as nn
import torch.nn.functional as F
from models import otsu
from entmax import sparsemax, entmax_bisect


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
        self.temperature = nn.Parameter(torch.ones(1))
        self.entmax_alpha = nn.Parameter(torch.tensor(1.5))  # learnable alpha for entmax

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
            A = (A - A.mean(dim=1, keepdim=True)) / (A.std(dim=1, keepdim=True, unbiased=True) + 1e-8)
            A = torch.sigmoid(A)  # sigmoid over K
        elif self.attention_activation == "min_max":
            A = (A - A.mean(dim=1, keepdim=True)) / (A.std(dim=1, keepdim=True, unbiased=True) + 1e-8)

            A_min = A.min(dim=1, keepdim=True)[0]
            A_max = A.max(dim=1, keepdim=True)[0]
            A = (A - A_min) / (A_max - A_min + 1e-8)  # min-max normalize to [0, 1]
        elif self.attention_activation == "softmax_temperature":
            A = torch.softmax(A / self.temperature.clamp(min=0.1), dim=1)  # softmax over K with temperature
        elif self.attention_activation == "sparsemax":
            A = sparsemax(A, dim=1)  # sparsemax over K
        elif self.attention_activation == "entmax":
            alpha_clamped = self.entmax_alpha.clamp(1.0, 2.0)
            A = entmax_bisect(A, alpha=alpha_clamped, dim=1) # entmax over K with learnable alpha
        else: # when not specified, default to softmax
            A = torch.softmax(A, dim=1)  # softmax over K

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
            if self.attention_activation == "softmax":
                threshold = 1.0 / K
            elif self.attention_activation == "sparsemax" or self.attention_activation == "entmax":
                threshold = 0.0  # sparsemax sets small values to exactly zero, so we can use zero as threshold
            else:
                threshold = otsu.compute_otsu_threshold(A)
        count = int((A.squeeze(0) > threshold).sum().item())
        return count, A, threshold
    
    def extract_features(self, x):
        """Gibt H (Feature-Vektoren) und A (Attention-Gewichte) zurück."""
        self.eval()
        with torch.no_grad():
            x = x.squeeze(0)
            H = self.feature_extractor_part1(x)
            H = H.view(-1, self.num_maps * self.pool_size * self.pool_size)
            H = self.feature_extractor_part2(H)  # [K, M]
            _, _, A = self.forward(x.unsqueeze(0))
        return H.cpu().numpy(), A.squeeze(0).cpu().numpy()

class AttentionBatchNorm(nn.Module):
    def __init__(self, M=500, L=128, num_maps=50, kernel_size=5, pool_size=4, ATTENTION_BRANCHES=1, in_channels=1, attention_activation="softmax"):
        super(AttentionBatchNorm, self).__init__()
        self.M = M
        self.L = L
        self.num_maps = num_maps
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.ATTENTION_BRANCHES = ATTENTION_BRANCHES
        self.attention_activation = attention_activation
        self.temperature = nn.Parameter(torch.ones(1))
        self.entmax_alpha = nn.Parameter(torch.tensor(1.5))  # learnable alpha for entmax

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=self.kernel_size, padding=self.kernel_size//2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=self.kernel_size, padding=self.kernel_size//2),
            nn.BatchNorm2d(64),
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
            A = (A - A.mean(dim=1, keepdim=True)) / (A.std(dim=1, keepdim=True, unbiased=True) + 1e-8)
            A = torch.sigmoid(A)  # sigmoid over K
        elif self.attention_activation == "min_max":
            A = (A - A.mean(dim=1, keepdim=True)) / (A.std(dim=1, keepdim=True, unbiased=True) + 1e-8)

            A_min = A.min(dim=1, keepdim=True)[0]
            A_max = A.max(dim=1, keepdim=True)[0]
            A = (A - A_min) / (A_max - A_min + 1e-8)  # min-max normalize to [0, 1]
        elif self.attention_activation == "softmax_temperature":
            A = torch.softmax(A / self.temperature.clamp(min=0.1), dim=1)  # softmax over K with temperature
        elif self.attention_activation == "sparsemax":
            A = sparsemax(A, dim=1)  # sparsemax over K
        elif self.attention_activation == "entmax":
            alpha_clamped = self.entmax_alpha.clamp(1.0, 2.0)
            A = entmax_bisect(A, alpha=alpha_clamped, dim=1) # entmax over K with learnable alpha
        else: # when not specified, default to softmax
            A = torch.softmax(A, dim=1)  # softmax over K

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
            if self.attention_activation == "softmax":
                threshold = 1.0 / K
            elif self.attention_activation == "sparsemax" or self.attention_activation == "entmax":
                threshold = 0.0  # sparsemax sets small values to exactly zero, so we can use zero as threshold
            else:
                threshold = otsu.compute_otsu_threshold(A)
        count = int((A.squeeze(0) > threshold).sum().item())
        return count, A, threshold
    
    def extract_features(self, x):
        """Gibt H (Feature-Vektoren) und A (Attention-Gewichte) zurück."""
        self.eval()
        with torch.no_grad():
            x = x.squeeze(0)
            H = self.feature_extractor_part1(x)
            H = H.view(-1, self.num_maps * self.pool_size * self.pool_size)
            H = self.feature_extractor_part2(H)  # [K, M]
            _, _, A = self.forward(x.unsqueeze(0))
        return H.cpu().numpy(), A.squeeze(0).cpu().numpy()
    
class AttentionThirdConv(nn.Module):
    def __init__(self, M=500, L=128, num_maps=50, kernel_size=5, pool_size=4, ATTENTION_BRANCHES=1, in_channels=1, attention_activation="softmax"):
        super(AttentionThirdConv, self).__init__()
        self.M = M
        self.L = L
        self.num_maps = num_maps
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.ATTENTION_BRANCHES = ATTENTION_BRANCHES
        self.attention_activation = attention_activation
        self.temperature = nn.Parameter(torch.ones(1))
        self.entmax_alpha = nn.Parameter(torch.tensor(1.5))  # learnable alpha for entmax

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=self.kernel_size, padding=self.kernel_size//2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=self.kernel_size, padding=self.kernel_size//2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=self.kernel_size//2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((self.pool_size, self.pool_size))
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(64 * self.pool_size * self.pool_size, self.M),
            nn.ReLU(),
            nn.Dropout(p=0.25)
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
            A = (A - A.mean(dim=1, keepdim=True)) / (A.std(dim=1, keepdim=True, unbiased=True) + 1e-8)
            A = torch.sigmoid(A)  # sigmoid over K
        elif self.attention_activation == "min_max":
            A = (A - A.mean(dim=1, keepdim=True)) / (A.std(dim=1, keepdim=True, unbiased=True) + 1e-8)

            A_min = A.min(dim=1, keepdim=True)[0]
            A_max = A.max(dim=1, keepdim=True)[0]
            A = (A - A_min) / (A_max - A_min + 1e-8)  # min-max normalize to [0, 1]
        elif self.attention_activation == "softmax_temperature":
            A = torch.softmax(A / self.temperature.clamp(min=0.1), dim=1)  # softmax over K with temperature
        elif self.attention_activation == "sparsemax":
            A = sparsemax(A, dim=1)  # sparsemax over K
        elif self.attention_activation == "entmax":
            alpha_clamped = self.entmax_alpha.clamp(1.0, 2.0)
            A = entmax_bisect(A, alpha=alpha_clamped, dim=1) # entmax over K with learnable alpha
        else: # when not specified, default to softmax
            A = torch.softmax(A, dim=1)  # softmax over K

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
            if self.attention_activation == "softmax":
                threshold = 1.0 / K
            elif self.attention_activation == "sparsemax" or self.attention_activation == "entmax":
                threshold = 0.0  # sparsemax sets small values to exactly zero, so we can use zero as threshold
            else:
                threshold = otsu.compute_otsu_threshold(A)
        count = int((A.squeeze(0) > threshold).sum().item())
        return count, A, threshold
    
    def extract_features(self, x):
        """Gibt H (Feature-Vektoren) und A (Attention-Gewichte) zurück."""
        self.eval()
        with torch.no_grad():
            x = x.squeeze(0)
            H = self.feature_extractor_part1(x)
            H = H.view(-1, self.num_maps * self.pool_size * self.pool_size)
            H = self.feature_extractor_part2(H)  # [K, M]
            _, _, A = self.forward(x.unsqueeze(0))
        return H.cpu().numpy(), A.squeeze(0).cpu().numpy()
    
class AttentionDropout(nn.Module):
    def __init__(self, M=500, L=128, num_maps=50, kernel_size=5, pool_size=4, ATTENTION_BRANCHES=1, in_channels=1, attention_activation="softmax"):
        super(AttentionThirdConv, self).__init__()
        self.M = M
        self.L = L
        self.num_maps = num_maps
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.ATTENTION_BRANCHES = ATTENTION_BRANCHES
        self.attention_activation = attention_activation
        self.temperature = nn.Parameter(torch.ones(1))
        self.entmax_alpha = nn.Parameter(torch.tensor(1.5))  # learnable alpha for entmax

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=self.kernel_size, padding=self.kernel_size//2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=self.kernel_size, padding=self.kernel_size//2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=self.kernel_size//2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((self.pool_size, self.pool_size))
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(64 * self.pool_size * self.pool_size, self.M),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Dropout(p=0.25),
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
            A = (A - A.mean(dim=1, keepdim=True)) / (A.std(dim=1, keepdim=True, unbiased=True) + 1e-8)
            A = torch.sigmoid(A)  # sigmoid over K
        elif self.attention_activation == "min_max":
            A = (A - A.mean(dim=1, keepdim=True)) / (A.std(dim=1, keepdim=True, unbiased=True) + 1e-8)

            A_min = A.min(dim=1, keepdim=True)[0]
            A_max = A.max(dim=1, keepdim=True)[0]
            A = (A - A_min) / (A_max - A_min + 1e-8)  # min-max normalize to [0, 1]
        elif self.attention_activation == "softmax_temperature":
            A = torch.softmax(A / self.temperature.clamp(min=0.1), dim=1)  # softmax over K with temperature
        elif self.attention_activation == "sparsemax":
            A = sparsemax(A, dim=1)  # sparsemax over K
        elif self.attention_activation == "entmax":
            alpha_clamped = self.entmax_alpha.clamp(1.0, 2.0)
            A = entmax_bisect(A, alpha=alpha_clamped, dim=1) # entmax over K with learnable alpha
        else: # when not specified, default to softmax
            A = torch.softmax(A, dim=1)  # softmax over K

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
            if self.attention_activation == "softmax":
                threshold = 1.0 / K
            elif self.attention_activation == "sparsemax" or self.attention_activation == "entmax":
                threshold = 0.0  # sparsemax sets small values to exactly zero, so we can use zero as threshold
            else:
                threshold = otsu.compute_otsu_threshold(A)
        count = int((A.squeeze(0) > threshold).sum().item())
        return count, A, threshold
    
    def extract_features(self, x):
        """Gibt H (Feature-Vektoren) und A (Attention-Gewichte) zurück."""
        self.eval()
        with torch.no_grad():
            x = x.squeeze(0)
            H = self.feature_extractor_part1(x)
            H = H.view(-1, self.num_maps * self.pool_size * self.pool_size)
            H = self.feature_extractor_part2(H)  # [K, M]
            _, _, A = self.forward(x.unsqueeze(0))
        return H.cpu().numpy(), A.squeeze(0).cpu().numpy()
    
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
        self.temperature = nn.Parameter(torch.ones(1))

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

        if self.attention_activation == "sigmoid" or self.attention_activation == "min_max":
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
            A = (A - A.mean(dim=1, keepdim=True)) / (A.std(dim=1, keepdim=True, unbiased=True) + 1e-8)
            A = torch.sigmoid(A)  # sigmoid over K
        elif self.attention_activation == "min_max":
            A = (A - A.mean(dim=1, keepdim=True)) / (A.std(dim=1, keepdim=True, unbiased=True) + 1e-8)

            A_min = A.min(dim=1, keepdim=True)[0]
            A_max = A.max(dim=1, keepdim=True)[0]
            A = (A - A_min) / (A_max - A_min + 1e-8)  # min-max normalize to [0, 1]
        elif self.attention_activation == "softmax_temperature":
            A = torch.softmax(A / self.temperature.clamp(min=0.1), dim=1)  # softmax over K with temperature
        else:
            A = torch.softmax(A, dim=1)  # softmax over K

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
        _, _, A = self.forward(X)
        K = A.shape[1]  # number of instances
        if threshold is None:
            if self.attention_activation == "softmax":
                threshold = 1.0 / K
            else:
                threshold = otsu.compute_otsu_threshold(A)
        count = int((A.squeeze(0) > threshold).sum().item())
        return count, A, threshold