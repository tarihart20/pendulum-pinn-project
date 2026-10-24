import torch
import torch.nn as nn

G = 9.81

class PendulumNN(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.correction = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, length_m):
        safe_length = torch.clamp(length_m, min=1e-6)
        theory = 2.0 * torch.pi * torch.sqrt(safe_length / G)
        x = torch.log(safe_length)
        correction = self.correction(x)
        return theory * torch.exp(correction)