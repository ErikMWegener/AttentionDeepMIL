import sys
import os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.learned_grayscale import LearnedGrayscale
import torch
import torch.nn as nn

class Backbone(nn.Module):
    def __init__(self, in_channels=3, kernel_size=3, num_maps=64, pool_size=4, M=500, grayscaling=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_maps = num_maps
        self.pool_size = pool_size
        self.M = M
        self.grayscaling = grayscaling

        self.grayscale_layer = LearnedGrayscale() if self.grayscaling else nn.Identity()
            
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

    def forward(self, x):
            x = x.squeeze(0)
            x = self.grayscale_layer(x)  # Apply learned grayscale conversion if enabled
    
            H = self.feature_extractor_part1(x)
            H = H.view(-1, self.num_maps * self.pool_size * self.pool_size)
            H = self.feature_extractor_part2(H)  # KxM

            return H