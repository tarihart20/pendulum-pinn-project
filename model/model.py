import torch
import torch.nn as nn

class PendulumNN(nn.Module):
    """
    Physics-informed model:
      T_hat(L) = T_theory(L) * exp(correction_NN(log(L)))

    This architecture keeps predictions positive and lets the neural network
    learn a multiplicative correction to T = 2*pi*sqrt(L/g).
    """
    def __init__(self, hidden=16):
        super().__init__()
        self.correction = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Start with approximately zero correction, i.e. the physical formula.
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, length_m):
        safe_length = torch.clamp(length_m, min=1e-6)
        theory = 2.0 * torch.pi * torch.sqrt(safe_length / G)
        # Scale length before feeding it to the NN.
        x = torch.log(safe_length)
        correction = self.correction(x)
        return theory * torch.exp(correction)