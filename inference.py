import torch
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffuser = hydra.utils.instantiate(cfg.diffuser).to(device)

    model = hydra.utils.instantiate(cfg.model).to(device)
    # model = load_checkpoint(model, cfg.ckpt_path, device)
    model.eval()
    latents = torch.randn(1, 4, 32, 32, device=device) # for ImageNet-256 + SD-VAE
    label = torch.randint(0, 1000, (1,), device=device)
    
    # normal sampling
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            diffuser.sample(
                unet=model,
                z=model,
                sample_steps=40,
                cfg_scale=1.5,
            )
            
    # TREAD sampling
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Models trained with TREAD extrapolate to unseen selection ratios (here gamma).
            # We can use this for GUIDED sampling.
            # For example, gamma1=0.3 and gamma2=0.7 are not seen during training.
            # The model can still sample with these values.
            # This enables models trained with TREAD to achieve a on-the-fly trade-off between
            # quality and FLOPS.
            # One can use the delta between gamma1 and gamma2 to guide the sampling,
            # but it can also be combined with CFG (class dropout) or AutoGuidance (unet1 != unet2).
            diffuser.sample_tread(
                unet1=model,
                gamma1=0.3,
                unet2=model,
                gamma2=0.7,
                z=latents,
                y=label,
                sample_steps=40,
                cfg_scale=1.5,
            )

if __name__ == "__main__":
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
