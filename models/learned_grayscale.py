import torch
import torch.nn as nn

class LearnedGrayscale(nn.Module):
    """Lernbarer Graustufen-Filter (RGB -> 1 Kanal) als nn.Conv2d mit 1x1-Kernel."""
    def __init__(self, init=(0.299, 0.587, 0.114)): #BT.601 standard conversion weights
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor(init).view(1, 3, 1, 1))

    @torch.no_grad()
    def normalized_weights(self):
        """L1-normalized |weights| in RGB order -> fractional channel contribution (sums to 1)."""
        w = self.conv.weight.detach().view(3)          # [w_r, w_g, w_b]
        return (w.abs() / (w.abs().sum() + 1e-8)).cpu()
    
    def forward(self, x):   
        return self.conv(x) 