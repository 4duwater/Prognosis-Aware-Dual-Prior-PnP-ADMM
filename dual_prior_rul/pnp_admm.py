import torch
import torch.nn.functional as F


def lowpass(x, width=5):
    z = F.pad(x.transpose(1, 2), (width//2, width//2), mode='replicate')
    return F.avg_pool1d(z, width, stride=1).transpose(1, 2)


def split_components(x):
    trend = lowpass(x)
    return trend, x - trend


def frequency_projectors(x, low_modes=6):
    n=x.shape[1]
    t=torch.arange(n,device=x.device,dtype=x.dtype).unsqueeze(1)
    k=torch.arange(n,device=x.device,dtype=x.dtype).unsqueeze(0)
    q=torch.cos(torch.pi*(t+0.5)*k/n)*(2/n)**0.5
    q[:,0]/=2**0.5
    pl=q[:,:low_modes]@q[:,:low_modes].T
    ph=torch.eye(n,device=x.device,dtype=x.dtype)-pl
    return pl,ph
