"""Pure-PyTorch Mamba-1 mixer, mirroring ``mamba_ssm.modules.mamba_simple.Mamba`` (v1).

Same parameter names, shapes, initialisation and ``_no_weight_decay`` flags, so weights can be
copied between this and the CUDA mixer. The selective scan is a plain sequential loop: correct
but slow. Used on CPU (tests, CI, smoke) and as an emergency fallback.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class MambaMixerRef(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16)
        d_inner = self.d_inner

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=bias)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv, groups=d_inner, padding=d_conv - 1, bias=conv_bias
        )
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)

        # dt_proj: weight ~ U(-r^-0.5, r^-0.5); bias = softplus^-1(dt), dt log-uniform in range
        std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -std, std)
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.dt_proj.bias._no_reinit = True  # type: ignore[attr-defined]

        # S4D-real initialisation: A = -[1, 2, ..., N] for every channel
        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]
        self.D = nn.Parameter(torch.ones(d_inner))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """[B, L, d_model] -> [B, L, d_model]. Causal: position t only sees positions <= t."""
        length = hidden_states.shape[1]
        x, z = self.in_proj(hidden_states).chunk(2, dim=-1)
        x = F.silu(self.conv1d(x.transpose(1, 2))[..., :length]).transpose(1, 2)
        dt, b, c = self.x_proj(x).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        a = -torch.exp(self.A_log.float())
        y = selective_scan_ref(x, dt, a, b, c, self.D)
        y = y * F.silu(z.float())
        return self.out_proj(y.to(hidden_states.dtype))


def selective_scan_ref(x, dt, a, b, c, d) -> torch.Tensor:
    """h_t = exp(dt_t A) h_{t-1} + dt_t B_t x_t;  y_t = C_t h_t + D x_t   (computed in fp32).

    Shapes: x, dt [B, L, Din]; a [Din, N]; b, c [B, L, N]; d [Din].
    """
    x, dt, b, c = x.float(), dt.float(), b.float(), c.float()
    d_a = torch.exp(dt.unsqueeze(-1) * a)  # [B, L, Din, N]
    d_bx = dt.unsqueeze(-1) * b.unsqueeze(2) * x.unsqueeze(-1)  # [B, L, Din, N]
    h = torch.zeros_like(d_a[:, 0])
    ys = []
    for t in range(x.shape[1]):
        h = d_a[:, t] * h + d_bx[:, t]
        ys.append((h * c[:, t].unsqueeze(1)).sum(-1))
    return torch.stack(ys, dim=1) + x * d.float()
