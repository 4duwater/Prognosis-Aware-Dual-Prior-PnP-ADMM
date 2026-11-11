import torch
from .core import temporal_loss
from .pnp_admm import lowpass,frequency_projectors
from .physics_layer import PhysicsLayer

VARIANTS={
 'none':('none','post',False),
 'legacy':('crop','post',False),
 'goal_only':('pair','post',False),
 'update_only':('crop','data',False),
 'goal_update':('pair','data',False),
 'goal_reliable':('pair','post',True),
 'full':('pair','data',True),
}

def mv(matrix,x):
    return (matrix@x.transpose(1,2).unsqueeze(-1)).squeeze(-1).transpose(1,2)
def dot(x,y):return (x*y).flatten(1).sum(1)

class Guidance:
    def __init__(self,rul,goal,anchor=None,valid=None,tolerance=0.):
        self.rul=rul;self.goal=goal;self.anchor=anchor;self.valid=valid;self.tolerance=tolerance
    def __call__(self,x):
        if self.goal=='crop':return temporal_loss(self.rul,x)
        current=self.rul(x).flatten()
        # Past window is exactly 30 cycles, ends 3 cycles earlier, reconstructed
        # without guidance from observed history. No gradient or label in anchor.
        drop=self.anchor-current;tau=self.tolerance
        return .5*(torch.relu(-drop-tau).square()+torch.relu(drop-3/125-tau).square())*self.valid
    def gradient(self,x):
        with torch.enable_grad():
            z=x.detach().requires_grad_(True);loss=self(z)
            gradient=torch.autograd.grad(loss.sum(),z)[0]
        return loss.detach(),gradient.detach()

class GuidedADMM:
    def __init__(self, curvature_variance, inner_steps=2, backtracks=10, trust_limit=.02):
        self.physics=PhysicsLayer(curvature_variance, mu=10.);self.inner_steps=inner_steps
        self.backtracks=backtracks;self.trust_limit=trust_limit;self.stats=[]
    def data_guidance(self,base_low,base_high,q1,q2,im1,im2,missing,guide,strength,iteration):
        """Joint optimization on missing entries before either prior and dual update.

        On missing coordinates the sequential unguided updates are a stationary
        point of both original data quadratics (M coupling is zero there).
        Thus their exact objective increment is .5 dl Q1 dl + .5 dh Q2 dh.
        Armijo updates reduce this increment plus strength*G(base+dl+dh).
        This is an inexact nonconvex data solve, not a convergence theorem.
        """
        base=base_low+base_high;dl=torch.zeros_like(base);dh=torch.zeros_like(base)
        for inner in range(self.inner_steps):
            loss,gradient=guide.gradient(base+dl+dh)
            gl=(mv(q1,dl)+strength*gradient)*missing
            gh=(mv(q2,dh)+strength*gradient)*missing
            pl=-mv(im1,gl);ph=-mv(im2,gh)
            # One common scale per sample preserves the preconditioned direction.
            maximum=(pl+ph).abs().flatten(1).max(1).values
            scale=(self.trust_limit/maximum.clamp_min(1e-20)).clamp_max(1)
            pl=pl*scale[:,None,None];ph=ph*scale[:,None,None]
            directional=dot(gl,pl)+dot(gh,ph)
            old=.5*dot(dl,mv(q1,dl))+.5*dot(dh,mv(q2,dh))+strength*loss
            alpha=torch.ones(len(base),device=base.device,dtype=base.dtype)
            accepted=torch.zeros(len(base),device=base.device,dtype=torch.bool)
            best_l=dl.clone();best_h=dh.clone();best_loss=old.clone()
            for backtrack in range(self.backtracks):
                trial_l=dl+alpha[:,None,None]*pl;trial_h=dh+alpha[:,None,None]*ph
                candidate=.5*dot(trial_l,mv(q1,trial_l))+.5*dot(trial_h,mv(q2,trial_h))+strength*guide(base+trial_l+trial_h)
                ok=(candidate<=old+1e-4*alpha*directional+1e-9)&(~accepted)
                best_l=torch.where(ok[:,None,None],trial_l,best_l)
                best_h=torch.where(ok[:,None,None],trial_h,best_h)
                best_loss=torch.where(ok,candidate,best_loss)
                accepted=accepted|ok
                if accepted.all().item():break
                alpha=torch.where(accepted,alpha,alpha*.5)
            self.stats.append(dict(iteration=iteration,inner=inner,
                objective_before_mean=old.mean().item(),objective_after_mean=best_loss.mean().item(),
                objective_increase_max=(best_loss-old).max().item(),
                accepted_fraction=accepted.float().mean().item(),backtracks=backtrack+1,
                step_rms=((best_l-dl+best_h-dh).square().mean().sqrt().item()),
                observed_delta_max=((best_l+best_h)*(1-missing)).abs().max().item(),
                direction_dot_gradient_max=directional.max().item()))
            dl=best_l;dh=best_h
        return base_low+dl,base_high+dh

    @torch.no_grad()
    def __call__(self,obs,mask,den,rul=None,variant='none',strength=0.,anchor=None,valid=None,tolerance=0.):
        goal,update,reliable=VARIANTS[variant]
        enabled=strength>0 and goal!='none' and bool((mask<1).any())
        guide=Guidance(rul,goal,anchor,valid,tolerance if reliable else 0.) if enabled else None
        self.physics.warm=None
        pl,ph=frequency_projectors(obs,6);rho=1.5
        trend=lowpass(obs)/lowpass(mask).clamp_min(1e-8)
        low=trend;high=(obs-trend)*mask;d=low;v=high
        u1=torch.zeros_like(obs);u2=torch.zeros_like(obs)
        diagonal=torch.diag_embed(mask.transpose(1,2)+rho)
        q1=diagonal+1.5*ph;q2=diagonal+1.5*pl
        inv1=torch.linalg.inv(q1);inv2=torch.linalg.inv(q2)
        missing=1-mask
        if enabled and update=='data':
            mt=missing.transpose(1,2)
            im1=torch.linalg.inv(q1*mt.unsqueeze(-1)*mt.unsqueeze(-2)+torch.diag_embed(mask.transpose(1,2)))
            im2=torch.linalg.inv(q2*mt.unsqueeze(-1)*mt.unsqueeze(-2)+torch.diag_embed(mask.transpose(1,2)))
        for iteration in range(10):
            low=mv(inv1,mask*(obs-high)+rho*(d-u1))
            high=mv(inv2,mask*(obs-low)+rho*(v-u2))
            if enabled and update=='data':
                low,high=self.data_guidance(low,high,q1,q2,im1,im2,missing,guide,strength,iteration)
            d=self.physics(low+u1);v=den(high+u2)
            if enabled and update=='post':
                _,gradient=guide.gradient(low+high)
                step=(strength*gradient).clamp(-.02,.02)*missing
                low=low-.5*step;high=high-.5*step
            u1=u1+low-d;u2=u2+high-v
        return low+high
