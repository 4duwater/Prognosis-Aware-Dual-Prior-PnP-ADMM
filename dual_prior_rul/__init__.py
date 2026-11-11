"""Dual-prior reconstruction with optional prediction guidance."""

import math

import torch
from torch import nn

from .core import RUL_SCALE, Reconstructor
from .guided_admm import GuidedADMM
from .model import PretrainTransformer
from .pnp_admm import split_components

__all__ = ["DualPriorReconstructor", "PretrainTransformer", "RUL_SCALE", "split_components"]


class DualPriorReconstructor(nn.Module):
    """Restore C-MAPSS windows of shape (batch, 30 cycles, 14 sensors)."""

    def __init__(self, curvature_variance):
        super().__init__()
        variance = torch.as_tensor(curvature_variance, dtype=torch.float64).detach().clone()
        if variance.shape != (14,) or not (torch.isfinite(variance) & (variance > 0)).all():
            raise ValueError("curvature_variance must contain 14 finite positive values")
        self.register_buffer("curvature_variance", variance, persistent=False)
        self.denoiser = PretrainTransformer(
            seq_size=30, patch_size=3, in_chans=14, embed_dim=64, depth=1, num_heads=4,
        )

    @staticmethod
    def _observations(x, mask):
        if x.ndim != 3 or x.shape[0] == 0 or x.shape[1:] != (30, 14):
            raise ValueError("Expected a nonempty (batch, 30, 14) input")
        if x.dtype not in (torch.float32, torch.float64):
            raise TypeError("Use float32 or float64 observations")
        if mask.shape != x.shape or not ((mask == 0) | (mask == 1)).all():
            raise ValueError("mask must match the input shape and contain only 0 or 1")
        mask = mask.to(device=x.device, dtype=x.dtype)
        return torch.where(mask.bool(), x, 0.0), mask

    def reconstruct(self, observed, mask, *, predictor=None, history=None,
                    history_mask=None, history_valid=None, guidance_strength=0.0):
        """Restore the input. With zero guidance this is differentiable."""
        observed, mask = self._observations(observed, mask)
        if not math.isfinite(guidance_strength) or guidance_strength < 0:
            raise ValueError("guidance_strength must be finite and nonnegative")

        def unguided():
            return Reconstructor(self.curvature_variance)(observed, mask, self.denoiser)

        if guidance_strength == 0 or not (mask == 0).any():
            return unguided()
        if not callable(predictor):
            raise ValueError("Guidance requires an external RUL predictor")
        if history is None or history_mask is None:
            raise ValueError("Guidance requires history and history_mask")
        if history.shape != observed.shape or history.device != observed.device or history.dtype != observed.dtype:
            raise ValueError("History must have the same shape, device and dtype as the current input")
        if history_valid is None:
            valid = observed.new_ones(observed.shape[0])
        else:
            valid = torch.as_tensor(history_valid, device=observed.device, dtype=observed.dtype)
            if valid.shape != (observed.shape[0],) or not ((valid == 0) | (valid == 1)).all():
                raise ValueError("history_valid must be a binary vector with one value per sample")
        if not valid.any():
            return unguided()
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Use torch.no_grad(), not torch.inference_mode(), with guidance")
        history, history_mask = self._observations(history, history_mask)

        def predict(x):
            output = predictor(x)
            if not isinstance(output, torch.Tensor) or output.shape not in ((len(x),), (len(x), 1)):
                raise ValueError("predictor must return one normalized RUL value per window")
            return output.reshape(-1)

        with torch.no_grad():
            history_solver = GuidedADMM(self.curvature_variance)
            restored_history = history_solver(history, history_mask, self.denoiser)
            anchor = predict(restored_history)
            return GuidedADMM(self.curvature_variance)(
                observed, mask, self.denoiser, predict,
                'goal_update', guidance_strength, anchor, valid, 0.0
            )

    def forward(self, observed, mask, **guidance):
        """Return the reconstructed sensor window."""
        return self.reconstruct(observed, mask, **guidance)
