"""Mamba-1 stack (README: "Models"): x = x + Dropout(Mixer(RMSNorm(x))), no separate MLP.

* ``impl``: ``cuda`` = mamba_ssm fused kernels; ``ref`` = pure PyTorch; ``auto`` = cuda on a
  GPU, ref on CPU. A GPU without working kernels is an ERROR, never a silent slow fallback.
* Initialisation follows ``mamba_ssm``'s own ``_init_weights`` (see ``MambaModel``).
* BiMamba (seq_cls / tok_cls): Mixer(x) + Reverse(Mixer(Reverse(x))) with SHARED weights.
* Padding: mamba_ssm has no padding mask. Batches are right-padded, pads are zeroed before each
  mixer, and Reverse is length-aware (only real tokens are reversed; pads stay at the end), so
  in both directions the scan reads every real token before any pad.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from mvt.core.config import MambaConfig, ScaleConfig
from mvt.core.env import mamba_ssm_import_error, mamba_ssm_version
from mvt.core.registry import register_model
from mvt.core.types import TaskSpec
from mvt.models.common import RMSNorm, SequenceModel
from mvt.models.mamba_ref import MambaMixerRef


def resolve_impl(impl: str, on_gpu: bool | None = None) -> str:
    """Pick the mixer implementation. Raises instead of silently running the slow scan on a GPU."""
    if on_gpu is None:
        on_gpu = torch.cuda.is_available()
    if impl == "auto":
        impl = "cuda" if on_gpu else "ref"
    if impl == "cuda":
        if not on_gpu:
            raise RuntimeError("Mamba impl 'cuda' needs a GPU")
        if not mamba_ssm_version():
            raise RuntimeError(
                f"GPU found but mamba_ssm does not import ({mamba_ssm_import_error()}). "
                "Install it as in the README (Troubleshooting). To run the slow reference scan "
                "on purpose, set `impl: ref` in configs/models/mamba.yaml."
            )
    return impl


def make_mixer(d_model: int, cfg: MambaConfig, impl: str) -> nn.Module:
    kwargs = dict(
        d_model=d_model,
        d_state=cfg.d_state,
        d_conv=cfg.d_conv,
        expand=cfg.expand,
        dt_min=cfg.dt_min,
        dt_max=cfg.dt_max,
        dt_init_floor=cfg.dt_init_floor,
        conv_bias=cfg.conv_bias,
        bias=cfg.bias,
    )
    if impl == "cuda":
        from mamba_ssm import Mamba  # lazy: CPU environments never import mamba_ssm

        return Mamba(**kwargs)
    return MambaMixerRef(**kwargs)


def reverse_padded(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Reverse each row's first ``lengths[b]`` positions; pads stay at the end. Self-inverse."""
    b, length = x.shape[:2]
    t = torch.arange(length, device=x.device).expand(b, length)
    n = lengths.unsqueeze(1)
    idx = torch.where(t < n, n - 1 - t, t)
    return x.gather(1, idx.unsqueeze(-1).expand_as(x))


class BiMixer(nn.Module):
    """Bidirectional mixer with one set of weights shared by both directions."""

    def __init__(self, mixer: nn.Module) -> None:
        super().__init__()
        self.mixer = mixer

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        backward = reverse_padded(self.mixer(reverse_padded(x, lengths)), lengths)
        return self.mixer(x) + backward


class MambaLayer(nn.Module):
    def __init__(
        self, d_model: int, cfg: MambaConfig, impl: str, bidirectional: bool, dropout: float
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.norm = RMSNorm(d_model, cfg.norm_eps)
        mixer = make_mixer(d_model, cfg, impl)
        self.mixer = BiMixer(mixer) if bidirectional else mixer
        self.drop = nn.Dropout(dropout)

    def core_mixer(self) -> nn.Module:
        return self.mixer.mixer if self.bidirectional else self.mixer

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.norm(x) * mask.unsqueeze(-1).to(x.dtype)  # zero the pads
        h = self.mixer(h, mask.sum(1)) if self.bidirectional else self.mixer(h)
        return x + self.drop(h)


@register_model("mamba")
class MambaModel(SequenceModel):
    def __init__(
        self, scale: ScaleConfig, model_cfg: MambaConfig, spec: TaskSpec, dropout: float
    ) -> None:
        impl = resolve_impl(model_cfg.impl)
        layers = [
            MambaLayer(scale.d_model, model_cfg, impl, not spec.causal, dropout)
            for _ in range(scale.n_layers)
        ]
        super().__init__(spec, scale.d_model, layers, dropout, model_cfg.norm_eps)
        self.mamba_impl = impl
        self._init_weights(model_cfg.init_std, scale.n_layers)
        self.tie_weights()

    def _init_weights(self, std: float, n_layers: int) -> None:
        """Same scheme as ``mamba_ssm.models.mixer_seq_simple._init_weights`` (Mamba-1):

        * mixer internals (A_log, D, dt_proj, conv1d) keep the mixer's own initialisation;
        * other linear weights keep PyTorch's default; their biases are zeroed unless marked
          ``_no_reinit`` (dt_proj.bias);
        * out_proj: kaiming-uniform, then divided by sqrt(n_layers) (one residual per layer);
        * embedding ~ N(0, std). The task head uses the same init as the Transformer's.
        """
        for layer in self.layers:
            mixer = layer.core_mixer()
            for proj in (mixer.in_proj, mixer.x_proj, mixer.dt_proj, mixer.out_proj):
                if proj.bias is not None and not getattr(proj.bias, "_no_reinit", False):
                    nn.init.zeros_(proj.bias)
            nn.init.kaiming_uniform_(mixer.out_proj.weight, a=math.sqrt(5))
            with torch.no_grad():
                mixer.out_proj.weight /= math.sqrt(n_layers)
        self.init_embedding_and_head(std)
