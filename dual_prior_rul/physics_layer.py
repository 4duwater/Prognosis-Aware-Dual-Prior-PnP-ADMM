import torch

class ProxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,z,solver):
        d,a1,a0,u1,u0,matrices=solver.solve(z)
        if ctx.needs_input_grad[0]:
            d1,qinv,mu=matrices
            active1=(mu*u1).abs()<solver.lam-1e-5
            active0=(mu*u0).abs()<2*solver.lam-1e-5
            groups=torch.cat([torch.zeros_like(active1[...,:1],dtype=torch.long),
                              (~active0).long().cumsum(-1)],-1)
            n=d1.shape[0]
            indices=torch.arange(n,device=z.device).expand_as(groups)
            anchor=torch.where(active1,indices,torch.full_like(indices,-1))
            prev=torch.cat([torch.full_like(anchor[...,:1],-1),anchor.cummax(-1).values[...,:-1]],-1)
            valid=active1&(prev>=0)&(groups==groups.gather(-1,prev.clamp_min(0)))
            constraints=(d1[indices]-d1[prev.clamp_min(0)])*valid.unsqueeze(-1)
            cq=constraints@qinv
            gram=cq@constraints.transpose(-1,-2)
            gram=gram+torch.diag_embed((~valid).to(z.dtype))
            correction=cq.transpose(-1,-2).double()@torch.linalg.solve(gram.double(),cq.double())
            jacobian=qinv-correction.to(z.dtype)
            ctx.save_for_backward(jacobian)
        return d.transpose(1,2)
    @staticmethod
    def backward(ctx,grad):
        (jacobian,)=ctx.saved_tensors
        out=(jacobian@grad.transpose(1,2).unsqueeze(-1)).squeeze(-1).transpose(1,2)
        return out,None

class PhysicsLayer:
    def __init__(self, variance, mu=10., max_iterations=10000, tolerance=1e-5):
        self.variance = variance
        self.mu=mu;self.max_iterations=max_iterations;self.tolerance=tolerance
        self.lam=.1/1.5;self.cache={};self.stats=[];self.warm=None
    def matrices(self,z):
        key=(z.device,z.dtype,z.shape[1],self.mu)
        if key not in self.cache:
            n=z.shape[1];eye=torch.eye(n,device=z.device,dtype=z.dtype)
            d1=eye[1:]-eye[:-1];d2=d1[1:]-d1[:-1]
            ew=torch.eye(n-1,device=z.device,dtype=z.dtype);dw=ew[1:]-ew[:-1]
            variance=torch.as_tensor(self.variance,device=z.device,dtype=z.dtype)
            q=eye+self.lam*(d2.T@d2)[None]/variance[:,None,None]
            qinv=torch.linalg.inv(q.double()).to(z.dtype)
            minv=torch.linalg.inv(ew+dw.T@dw)
            a=d1.T@minv
            system=q+self.mu*(d1.T@d1-a@d1)
            inverse=torch.linalg.inv(system.double()).to(z.dtype)
            self.cache[key]=(d1,dw,minv,a,inverse,qinv)
        return self.cache[key]
    def solve(self,z):
        d1,dw,minv,a,inverse,qinv=self.matrices(z)
        x=z.transpose(1,2);d=x.clone();w=torch.zeros_like(x[...,:-1])
        a1=d@d1.T-w;a0=w@dw.T;u1=torch.zeros_like(a1);u0=torch.zeros_like(a0)
        if self.warm is not None:
            w,a1,a0,u1,u0=self.warm
        mu=self.mu
        for step in range(self.max_iterations):
            rhsw=-(a1-u1)+(a0-u0)@dw
            rhs=x+mu*((a1-u1)@d1+rhsw@a.T)
            d=(rhs.transpose(0,1)@inverse.transpose(-1,-2)).transpose(0,1)
            w=(rhsw+d@d1.T)@minv.T
            old1=a1;old0=a0
            k1=d@d1.T-w;k0=w@dw.T
            relaxed1=1.8*k1-.8*old1;relaxed0=1.8*k0-.8*old0
            r1=relaxed1+u1;r0=relaxed0+u0
            a1=r1.sign()*(r1.abs()-self.lam/mu).clamp_min(0)
            a0=r0.sign()*(r0.abs()-2*self.lam/mu).clamp_min(0)
            residual1=k1-a1;residual0=k0-a0
            u1=u1+relaxed1-a1;u0=u0+relaxed0-a0
            if (step+1)%50==0 or step+1==self.max_iterations:
                primal=max(residual1.abs().max().item(),residual0.abs().max().item())
                dual=max((mu*(a1-old1)@d1).abs().max().item(),
                         (mu*(-(a1-old1)+(a0-old0)@dw)).abs().max().item())
                if primal<self.tolerance and dual<2*self.tolerance:break
        self.stats.append(dict(iterations=step+1,primal_max=primal,dual_max=dual))
        self.warm=(w.detach(),a1.detach(),a0.detach(),u1.detach(),u0.detach())
        return d,a1,a0,u1,u0,(d1,qinv,mu)
    def __call__(self,z):
        return ProxFunction.apply(z,self)
