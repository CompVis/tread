import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
from tqdm import tqdm
from functools import partial
from utils.edm_helper import *
from autoencoder import SDVAE_EMA

def edm_sampler(
        net, latents, class_labels=None, cfg_scale=None, feat=None, randn_like=torch.randn_like,
        num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
        S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat.float(), t_hat, class_labels.long(), cfg_scale, feat=feat).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next.float(), t_next, class_labels.long(), cfg_scale, feat=feat).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

class EDMPrecond(nn.Module):
    def __init__(self,
                 img_resolution,
                 img_channels,
                 num_classes=0,
                 sigma_min=0,
                 sigma_max=float('inf'),
                 sigma_data=0.5,
                 model=None,
                 ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = model

    def forward(self, x, sigma, class_labels=None, cfg_scale=None, **model_kwargs):
        model_fn = self.model if cfg_scale is None else partial(self.model.forward_with_cfg, cfg_scale=cfg_scale)
        sigma = sigma.to(x.dtype).reshape(-1, 1, 1, 1)

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        
        F_x = model_fn(
            x=(c_in * x).to(x.dtype), 
            t=c_noise.flatten(), 
            y=class_labels,
            **model_kwargs
            )
        D_x = c_skip * x + c_out * F_x
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

class EDMDiffusion(nn.Module):
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, sigma_min=0, sigma_max=float('inf')):
        super().__init__()
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        num_gpus = torch.cuda.device_count()
        self.use_distributed = num_gpus > 1
        self.sampler_fn = edm_sampler
        
    def wrap_model_with_precond(self, model):
        precond = EDMPrecond(img_resolution=model.input_size, img_channels=model.in_channels,
                                num_classes=model.num_classes, sigma_min=self.sigma_min, sigma_max=self.sigma_max,
                                sigma_data=self.sigma_data, model=model)
        return precond

    def forward(self, model, x, sigma, y, cfg_scale=None, **model_kwargs):
        model_out = model(x, sigma, y, cfg_scale=cfg_scale, **model_kwargs)
        return model_out
    
    def get_training_loss(self, net, x, y=None, class_drop_prob=0.1):
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(x, dtype=x.dtype) * sigma
        D_yn = net(x + n, sigma, y, class_drop_prob=class_drop_prob)
        loss = weight * ((D_yn - x) ** 2)
        return loss
    
    @torch.no_grad()
    def generate(self, cfg, net, device, rank, size, outdir):
        seeds = parse_int_list(cfg.seeds)[:cfg.fid_num_samples]
        raw_net = unwrap_model(net)
        in_channels = raw_net.model.in_channels
        input_size = raw_net.model.input_size
        num_classes = raw_net.model.num_classes
        
        
        num_batches = ((len(seeds) - 1) // (cfg.max_batch_size * size) + 1) * size
        all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
        rank_batches = all_batches[rank:: size]

        net.eval()

        sampler_kwargs = dict(num_steps=cfg.num_steps, S_churn=cfg.S_churn,
                            solver=cfg.solver, discretization=cfg.discretization,
                            schedule=cfg.schedule, scaling=cfg.scaling)
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
        print(f"sampler_kwargs: {sampler_kwargs}, \nsampler fn: {self.sampler_fn.__name__}")
        vae = SDVAE_EMA().to(device)

        num_gpus = torch.cuda.device_count()
        use_distributed = num_gpus > 1
        for batch_seeds in tqdm(rank_batches, unit='batch', disable=(rank != 0)):
            batch_size = len(batch_seeds)
            if batch_size == 0:
                continue

            rnd = StackedRandomGenerator(device, batch_seeds)
            latents = rnd.randn([batch_size, in_channels, input_size, input_size], device=device)
            if num_classes:
                class_labels = rnd.randint(0, num_classes, size=[batch_size], device=device)

            if cfg.class_idx is not None:
                class_labels[:, :] = 0
                class_labels[:, cfg.class_idx] = 1

            feat = None
            
            def recur_decode(z):
                try:
                    return vae.decode(z)
                except:
                    assert z.shape[2] % 2 == 0
                    z1, z2 = z.tensor_split(2)
                    return torch.cat([recur_decode(z1), recur_decode(z2)])
            with torch.no_grad():
                z = self.sampler_fn(net, latents.float(), class_labels.float(), randn_like=rnd.randn_like,
                            cfg_scale=cfg.cfg_scale, feat=feat, **sampler_kwargs).float()
                images = recur_decode(z)
                
            images_np = images.cpu().numpy()
            for seed, image_np in zip(batch_seeds, images_np):
                image_dir = os.path.join(outdir, f'{seed - seed % 1000:06d}') if cfg.subdirs else outdir
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, f'{seed:06d}.png')
                if image_np.shape[2] == 1:
                    PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
                else:
                    PIL.Image.fromarray(image_np, 'RGB').save(image_path)         