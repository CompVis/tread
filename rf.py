import re
import torch
from torch import nn

"""
The class structure is inspired by: https://github.com/cloneofsimo/minRF/blob/main/rf.py
"""

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

class RF(nn.Module):
    def __init__(self, train_timestep_sampling: str = "logit_sigmoid", immiscible: bool = False, dts_lambda: float = 0.0):
        super().__init__()
        self.train_timestep_sampling = train_timestep_sampling

    def forward(self, unet: nn.Module, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B = x.size(0)
        if self.train_timestep_sampling == "logit_sigmoid":
            t = torch.sigmoid(torch.randn(B, device=x.device))
        elif self.train_timestep_sampling == "uniform":
            t = torch.rand(B, device=x.device)
        else:
            raise ValueError(f'Unknown train timestep sampling method "{self.train_timestep_sampling}".')
        t_exp = t.view([B] + [1] * (x.ndim - 1))
        z1 = torch.randn_like(x)
        zt = (1 - t_exp) * x + t_exp * z1
        vtheta = unet(zt, t, **kwargs)
        loss = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, x.ndim)))
        return loss

    def sample(
        self,
        unet: nn.Module,
        z: torch.Tensor,
        sample_steps: int = 50,
        cfg_scale: float = 1.0,
        **kwargs,
    ):
        B = z.size(0)
        dt_value = 1.0 / sample_steps
        dt = torch.full((B,), dt_value, device=z.device, dtype=z.dtype) \
                .view([B] + [1] * (z.ndim - 1))

        for i in range(sample_steps, 0, -1):
            t = torch.full((B,), i / sample_steps, device=z.device, dtype=z.dtype)

            do_guidance = cfg_scale > 1.0
            do_unconditional_only = cfg_scale == 0.0
            
            if do_unconditional_only:
                vtheta = unet(z, t, class_drop_prob=1.0, **kwargs)
            else:
                vtheta = unet(z, t, class_drop_prob=0.0, force_dfs=do_guidance, **kwargs)
            
            if do_guidance:
                vtheta_uncond = unet(z, t, class_drop_prob=1.0, force_dfs=do_guidance, **kwargs)
                vtheta = vtheta_uncond + cfg_scale * (vtheta - vtheta_uncond)

            z = z - dt * vtheta
        return z
    
    def sample_tread(
        self, 
        unet1: nn.Module, 
        gamma1: float, 
        unet2: nn.Module, 
        gamma2: float, 
        z: torch.Tensor,
        sample_steps: int = 50,
        cfg_scale: float = 1.5,
        **kwargs
    ):
        B = z.size(0)
        dt_value = 1.0 / sample_steps
        dt = torch.full((B,), dt_value, device=z.device, dtype=z.dtype) \
                .view([B] + [1] * (z.ndim - 1))

        for i in range(sample_steps, 0, -1):
            t = torch.full((B,), i / sample_steps, device=z.device, dtype=z.dtype)

            assert cfg_scale > 1.0, "We assume cfg_scale > 1.0 for this TREAD sampling."          
            vtheta1 = unet1(z, t, class_drop_prob=1.0, force_routing=True, overwrite_selection_ratio=gamma1, **kwargs)
            vtheta2 = unet2(z, t, class_drop_prob=1.0, force_routing=True, overwrite_selection_ratio=gamma2, **kwargs)
            vtheta = vtheta2 + cfg_scale * (vtheta1 - vtheta2)
            z = z - dt * vtheta
            
        return z

class LatentRF(RF):
    def __init__(self, ae: nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.ae = ae

    def forward(self, unet: nn.Module, x: torch.Tensor, precomputed: bool = False, **kwargs) -> torch.Tensor:
        if not precomputed:
            with torch.no_grad():
                x = self.ae.encode(x)
        return super().forward(unet, x, **kwargs)

    def sample(self, unet: nn.Module, z: torch.Tensor, sample_steps: int = 50, return_list: bool = False, **kwargs):
        latent = super().sample(unet, z, sample_steps=sample_steps, return_list=return_list, **kwargs)
        return self.ae.decode(latent)