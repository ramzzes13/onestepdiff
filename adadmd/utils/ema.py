"""Exponential Moving Average for model parameters."""

import torch
import torch.nn as nn
from copy import deepcopy


class EMA:
    """Exponential Moving Average wrapper for model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)

    def forward(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


class EMAScalar:
    """EMA tracker for scalar values (used for adaptive weighting)."""

    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.value = None

    def update(self, new_value: float):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.decay * self.value + (1 - self.decay) * new_value
        return self.value

    def get(self, default: float = 1.0):
        return self.value if self.value is not None else default
