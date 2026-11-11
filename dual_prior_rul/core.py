# -*- coding: utf-8 -*-
"""Differentiable reconstruction and temporal loss for optional guidance."""
import torch
from .pnp_admm import lowpass,frequency_projectors
from .physics_layer import PhysicsLayer

RUL_SCALE = 125.0


def temporal_loss(rul,x):
    # Prefix ends three cycles earlier; suffix and full window share endpoint.
    full=rul(x).flatten();early=rul(x[:,:-3]).flatten();late=rul(x[:,3:]).flatten()
    return .5*(full-late).square()+.5*torch.relu(late-early).square()+.5*torch.relu(early-late-3/125).square()

class Reconstructor:
    def __init__(self, curvature_variance):
        self.physics = PhysicsLayer(curvature_variance, mu=10.)
    def __call__(self,obs,mask,den,guide=None,strength=0.,differentiate=False):
        self.physics.warm=None
        pl,ph=frequency_projectors(obs,6);rho=1.5
        trend=lowpass(obs)/lowpass(mask).clamp_min(1e-8)
        low=trend;high=(obs-trend)*mask
        d=low;v=high;u1=torch.zeros_like(obs);u2=torch.zeros_like(obs)
        diagonal=torch.diag_embed(mask.transpose(1,2)+rho)
        inv1=torch.linalg.inv(diagonal+1.5*ph);inv2=torch.linalg.inv(diagonal+1.5*pl)
        for iteration in range(10):
            rhs=mask*(obs-high)+rho*(d-u1)
            low=(inv1@rhs.transpose(1,2).unsqueeze(-1)).squeeze(-1).transpose(1,2)
            d=self.physics(low+u1)
            rhs=mask*(obs-low)+rho*(v-u2)
            high=(inv2@rhs.transpose(1,2).unsqueeze(-1)).squeeze(-1).transpose(1,2)
            v=den(high+u2)
            if guide is not None and strength>0:
                with torch.enable_grad():
                    signal=(low+high).detach().requires_grad_(True)
                    gradient=torch.autograd.grad(temporal_loss(guide,signal).sum(),signal)[0]
                step=(strength*gradient).clamp(-.02,.02)*(1-mask)
                low=low-.5*step;high=high-.5*step
            u1=u1+low-d;u2=u2+high-v
        return low+high
