"""LoRA (Low-Rank Adaptation) for the fake score model.

Instead of maintaining a full copy of the diffusion model for fake score estimation,
we use a LoRA adapter on the frozen base model.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional


class LoRALinear(nn.Module):
    """LoRA adapter for a linear layer: W' = W + BA where B(r,out), A(in,r)."""

    def __init__(self, in_features: int, out_features: int, rank: int = 4,
                 alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features) -> (..., out_features)
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LoRAConv2d(nn.Module):
    """LoRA adapter for Conv2d: applies low-rank update to convolution weights."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Decompose: Conv(in->out, k) ≈ Conv(in->r, 1) then Conv(r->out, k)
        self.lora_A = nn.Conv2d(in_channels, rank, 1, bias=False)
        self.lora_B = nn.Conv2d(rank, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(x)) * self.scaling


class LoRAWrapper(nn.Module):
    """Wraps a frozen base model with LoRA adapters.

    Applies LoRA to attention q, k, v, out projections (Conv2d layers named
    accordingly in the UNet's AttentionBlock).
    """

    def __init__(self, base_model: nn.Module, rank: int = 4, alpha: float = 1.0,
                 target_modules: Optional[list] = None):
        super().__init__()
        self.base_model = base_model
        # Freeze base model
        for p in self.base_model.parameters():
            p.requires_grad_(False)

        self.rank = rank
        self.alpha = alpha
        self.lora_layers: Dict[str, nn.Module] = nn.ModuleDict()

        if target_modules is None:
            target_modules = ['q', 'k', 'v', 'out']

        # Find and wrap target modules
        self._inject_lora(target_modules)

    def _inject_lora(self, target_modules: list):
        """Find attention layers and inject LoRA adapters."""
        for name, module in self.base_model.named_modules():
            if any(name.endswith(f'.{t}') for t in target_modules):
                if isinstance(module, nn.Conv2d):
                    lora = LoRAConv2d(
                        module.in_channels, module.out_channels,
                        module.kernel_size[0],
                        rank=self.rank, alpha=self.alpha
                    )
                    # Store with sanitized name (dots replaced)
                    key = name.replace('.', '_')
                    self.lora_layers[key] = lora
                elif isinstance(module, nn.Linear):
                    lora = LoRALinear(
                        module.in_features, module.out_features,
                        rank=self.rank, alpha=self.alpha
                    )
                    key = name.replace('.', '_')
                    self.lora_layers[key] = lora

        # Store mapping from module name to lora key
        self._name_to_key = {}
        for name, module in self.base_model.named_modules():
            if any(name.endswith(f'.{t}') for t in ['q', 'k', 'v', 'out']):
                key = name.replace('.', '_')
                if key in self.lora_layers:
                    self._name_to_key[name] = key

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """Forward pass: base model output + LoRA corrections.

        For efficiency, we hook into the base model's forward pass.
        """
        # Register hooks to add LoRA outputs
        hooks = []
        lora_outputs = {}

        def make_hook(lora_key):
            def hook_fn(module, input, output):
                lora_out = self.lora_layers[lora_key](input[0])
                return output + lora_out
            return hook_fn

        for name, module in self.base_model.named_modules():
            if name in self._name_to_key:
                key = self._name_to_key[name]
                h = module.register_forward_hook(make_hook(key))
                hooks.append(h)

        # Run forward pass with hooks active
        output = self.base_model(x, t, y)

        # Remove hooks
        for h in hooks:
            h.remove()

        return output

    def lora_parameters(self):
        """Return only the LoRA parameters (for optimizer)."""
        return self.lora_layers.parameters()

    def lora_state_dict(self):
        """Return only LoRA state dict."""
        return self.lora_layers.state_dict()

    def load_lora_state_dict(self, state_dict):
        self.lora_layers.load_state_dict(state_dict)

    def increase_rank(self, new_rank: int):
        """Increase LoRA rank (rank warming schedule).

        Creates new LoRA layers with higher rank, initialized from current weights.
        """
        if new_rank <= self.rank:
            return

        old_state = self.lora_state_dict()
        old_rank = self.rank
        self.rank = new_rank

        for key, lora in list(self.lora_layers.items()):
            if isinstance(lora, LoRAConv2d):
                in_ch = lora.lora_A.in_channels
                out_ch = lora.lora_B.out_channels
                ks = lora.lora_B.kernel_size[0]
                new_lora = LoRAConv2d(in_ch, out_ch, ks, rank=new_rank, alpha=self.alpha)
                # Copy old weights into first r columns/rows
                with torch.no_grad():
                    new_lora.lora_A.weight[:old_rank] = lora.lora_A.weight
                    new_lora.lora_B.weight[:, :old_rank] = lora.lora_B.weight
                self.lora_layers[key] = new_lora.to(lora.lora_A.weight.device)
            elif isinstance(lora, LoRALinear):
                in_f = lora.lora_A.shape[1]
                out_f = lora.lora_B.shape[0]
                new_lora = LoRALinear(in_f, out_f, rank=new_rank, alpha=self.alpha)
                with torch.no_grad():
                    new_lora.lora_A[:old_rank] = lora.lora_A
                    new_lora.lora_B[:, :old_rank] = lora.lora_B
                self.lora_layers[key] = new_lora.to(lora.lora_A.device)

    def num_lora_params(self):
        """Count trainable LoRA parameters."""
        return sum(p.numel() for p in self.lora_layers.parameters())
