"""Attention-based MIL models (AbMIL, Ilse et al. 2018) used in the thesis.

All models take one bag of K patches, compute one feature vector per patch,
weight the patches with an attention mechanism and classify the weighted sum.
Besides the bag prediction they expose ``count_positive_instances``, which turns
the attention weights into a patch count (thesis section "Counting").

Classes and where they appear in the thesis (section "Attention-based MIL"):

    Attention           standard AbMIL with the backbone from ``models/backbone.py``
    AttentionBatchNorm  backbone with GroupNorm after each conv layer
    AttentionThirdConv  GroupNorm + third conv layer + dropout after the FC layer
    AttentionDropout    as AttentionThirdConv, plus dropout in the attention branch
    GatedAttention      gated attention mechanism (Ilse et al. 2018)

All classes support the attention activations softmax (default), sigmoid,
min_max, sparsemax, entmax and softmax_temperature (``--attention_activation``).
"""
from models.backbone import Backbone
from models.learned_grayscale import LearnedGrayscale
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import otsu
from entmax import sparsemax, entmax_bisect


class Attention(nn.Module):
    """Standard attention-based MIL model (Ilse et al. 2018).

    Args:
        M: size of the patch feature vector produced by the backbone.
        L: hidden size of the attention network.
        num_maps, kernel_size, pool_size: backbone dimensions (see ``Backbone``).
        ATTENTION_BRANCHES: number of attention heads (always 1 in the thesis).
        in_channels: 1 for grayscale bags, 3 for colour bags.
        grayscaling: prepend the learnt grayscale conversion (``LearnedGrayscale``).
        attention_activation: normalisation of the attention scores, see module docstring.

    ``forward`` returns ``(Y_prob, Y_hat, A)``: bag probability, hard bag label and
    the attention weights with shape [ATTENTION_BRANCHES, K].
    """
    def __init__(self, M=500, L=128, num_maps=50, kernel_size=5, pool_size=4, ATTENTION_BRANCHES=1, in_channels=1, grayscaling=False, attention_activation="softmax"):
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
        self.grayscaling = grayscaling
        

        self.backbone = Backbone(in_channels=in_channels, kernel_size=kernel_size, num_maps=num_maps, pool_size=pool_size, M=M, grayscaling=grayscaling)

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
        H = self.backbone(x)  # KxM

        A = self.attention(H)  # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        # sigmoid and min_max judge every patch on its own: the raw scores are
        # standardised within the bag first, so the activation does not saturate.
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

        # Without softmax the weights do not sum to one, so the scale of Z would
        # grow with the bag size. Dividing by the mean weight and applying
        # LayerNorm restores the size invariance (thesis section "Attention-based MIL").
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
            threshold: Attention weight threshold. If None, the default of the
                activation is used: 1/K for softmax (uniform attention, K = number
                of instances), 0 for sparsemax/entmax and a per-bag Otsu threshold
                for all other activations.

        Returns:
            count: Number of instances with attention weight above the threshold.
            attention_weights: The attention weights for all instances.
            threshold: The threshold that was applied.
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
            H = self.backbone(x)
            _, _, A = self.forward(x.unsqueeze(0))
        return H.cpu().numpy(), A.squeeze(0).cpu().numpy()

class AttentionBatchNorm(nn.Module):
    """AbMIL with GroupNorm after each convolution ("group normalisation" in the thesis).

    Despite the class name, GroupNorm (4 groups) is used instead of BatchNorm,
    because a batch holds a single bag. The backbone is defined inline with
    32/64 feature maps; ``num_maps`` is ignored (fixed to 64). Attention head,
    activations and counting are identical to ``Attention``.
    """
    def __init__(self, M=500, L=128, num_maps=50, kernel_size=5, pool_size=4, ATTENTION_BRANCHES=1, in_channels=1, attention_activation="softmax"):
        super(AttentionBatchNorm, self).__init__()
        self.M = M
        self.L = L
        self.num_maps = 64
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.ATTENTION_BRANCHES = ATTENTION_BRANCHES
        self.attention_activation = attention_activation
        self.temperature = nn.Parameter(torch.ones(1))
        self.entmax_alpha = nn.Parameter(torch.tensor(1.5))  # learnable alpha for entmax

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=self.kernel_size, padding=self.kernel_size//2),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=self.kernel_size, padding=self.kernel_size//2),
            nn.GroupNorm(num_groups=4, num_channels=64),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((self.pool_size, self.pool_size))
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(64 * self.pool_size * self.pool_size, self.M),
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
        H = H.view(-1, 64 * self.pool_size * self.pool_size)
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
            threshold: Attention weight threshold. If None, the default of the
                activation is used: 1/K for softmax (uniform attention, K = number
                of instances), 0 for sparsemax/entmax and a per-bag Otsu threshold
                for all other activations.

        Returns:
            count: Number of instances with attention weight above the threshold.
            attention_weights: The attention weights for all instances.
            threshold: The threshold that was applied.
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
            H = H.view(-1, 64 * self.pool_size * self.pool_size)
            H = self.feature_extractor_part2(H)  # [K, M]
            _, _, A = self.forward(x.unsqueeze(0))
        return H.cpu().numpy(), A.squeeze(0).cpu().numpy()
    
class AttentionThirdConv(nn.Module):
    """AbMIL with a deeper backbone: GroupNorm, a third 3x3 convolution and dropout
    (p=0.25) after the fully connected layer. ``num_maps`` is ignored (fixed to 64).
    Attention head, activations and counting are identical to ``Attention``.
    """
    def __init__(self, M=500, L=128, num_maps=50, kernel_size=5, pool_size=4, ATTENTION_BRANCHES=1, in_channels=1, attention_activation="softmax"):
        super(AttentionThirdConv, self).__init__()
        self.M = M
        self.L = L
        self.num_maps = 64
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.ATTENTION_BRANCHES = ATTENTION_BRANCHES
        self.attention_activation = attention_activation
        self.temperature = nn.Parameter(torch.ones(1))
        self.entmax_alpha = nn.Parameter(torch.tensor(1.5))  # learnable alpha for entmax

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=self.kernel_size, padding=self.kernel_size//2),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=self.kernel_size, padding=self.kernel_size//2),
            nn.GroupNorm(num_groups=4, num_channels=64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=self.kernel_size//2),
            nn.GroupNorm(num_groups=4, num_channels=64),
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
        H = H.view(-1, 64 * self.pool_size * self.pool_size)
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
            threshold: Attention weight threshold. If None, the default of the
                activation is used: 1/K for softmax (uniform attention, K = number
                of instances), 0 for sparsemax/entmax and a per-bag Otsu threshold
                for all other activations.

        Returns:
            count: Number of instances with attention weight above the threshold.
            attention_weights: The attention weights for all instances.
            threshold: The threshold that was applied.
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
            H = H.view(-1, 64 * self.pool_size * self.pool_size)
            H = self.feature_extractor_part2(H)  # [K, M]
            _, _, A = self.forward(x.unsqueeze(0))
        return H.cpu().numpy(), A.squeeze(0).cpu().numpy()
    
class AttentionDropout(nn.Module):
    """Like ``AttentionThirdConv``, with an additional dropout layer (p=0.25) in the
    attention network. ``num_maps`` is ignored (fixed to 64).
    """
    def __init__(self, M=500, L=128, num_maps=50, kernel_size=5, pool_size=4, ATTENTION_BRANCHES=1, in_channels=1, attention_activation="softmax"):
        super(AttentionDropout, self).__init__()
        self.M = M
        self.L = L
        self.num_maps = 64
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.ATTENTION_BRANCHES = ATTENTION_BRANCHES
        self.attention_activation = attention_activation
        self.temperature = nn.Parameter(torch.ones(1))
        self.entmax_alpha = nn.Parameter(torch.tensor(1.5))  # learnable alpha for entmax

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=self.kernel_size, padding=self.kernel_size//2),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=self.kernel_size, padding=self.kernel_size//2),
            nn.GroupNorm(num_groups=4, num_channels=64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=self.kernel_size//2),
            nn.GroupNorm(num_groups=4, num_channels=64),
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
        H = H.view(-1, 64 * self.pool_size * self.pool_size)
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
            threshold: Attention weight threshold. If None, the default of the
                activation is used: 1/K for softmax (uniform attention, K = number
                of instances), 0 for sparsemax/entmax and a per-bag Otsu threshold
                for all other activations.

        Returns:
            count: Number of instances with attention weight above the threshold.
            attention_weights: The attention weights for all instances.
            threshold: The threshold that was applied.
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
            H = H.view(-1, 64 * self.pool_size * self.pool_size)
            H = self.feature_extractor_part2(H)  # [K, M]
            _, _, A = self.forward(x.unsqueeze(0))
        return H.cpu().numpy(), A.squeeze(0).cpu().numpy()
    
class GatedAttention(nn.Module):
    """Gated attention-based MIL (Ilse et al. 2018): the attention score is
    ``w^T (tanh(V h) * sigmoid(U h))``. Uses the same two-layer convolutional
    feature extractor as ``Backbone``, defined inline (no learnt grayscale layer).
    """
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
            threshold: Attention weight threshold. If None, the default of the
                activation is used: 1/K for softmax (uniform attention, K = number
                of instances) and a per-bag Otsu threshold for all other activations.

        Returns:
            count: Number of instances with attention weight above the threshold.
            attention_weights: The attention weights for all instances.
            threshold: The threshold that was applied.
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