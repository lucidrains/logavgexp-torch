import math
import torch
from torch import nn
import torch.nn.functional as F

def exists(t):
    return t is not None

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def l2norm(t):
    return F.normalize(t, dim = -1)

def logavgexp(
    t,
    mask = None,
    dim = -1,
    eps = 1e-20,
    temp = 0.01,
    keepdim = False
):
    if exists(mask):
        mask_value = -torch.finfo(t.dtype).max
        t = t.masked_fill(~mask, mask_value)
        n = mask.sum(dim = dim)
        norm = torch.log(n)
    else:
        n = t.shape[dim]
        norm = math.log(n)

    t = t / temp
    max_t = t.amax(dim = dim).detach()
    t_exp = (t - max_t.unsqueeze(dim)).exp()
    avg_exp = t_exp.sum(dim = dim).clamp(min = eps) / n
    out = log(avg_exp, eps = eps) + max_t - norm
    out = out * temp

    out = out.unsqueeze(dim) if keepdim else out
    return out

class LogAvgExp(nn.Module):
    def __init__(
        self,
        dim = -1,
        eps = 1e-20,
        temp = 0.01,
        keepdim = False,
        learned_temp = False
    ):
        super().__init__()
        assert temp >= 0 and temp <= 1., 'temperature must be between 0 and 1'

        self.learned_temp = learned_temp

        if learned_temp:
            self.temp = nn.Parameter(torch.ones((1,)) * math.log(temp))
        else:
            self.temp = temp

        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x, mask = None, eps = 1e-8):
        if not self.learned_temp:
            temp = self.temp
        else:
            temp = self.temp.exp().clamp(min = eps)

        return logavgexp(
            x,
            mask = mask,
            dim = self.dim,
            temp = temp,
            keepdim = self.keepdim
        )
