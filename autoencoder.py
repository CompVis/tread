import torch
import torch.nn as nn
from diffusers import AutoencoderKL

class SDVAE_EMA(nn.Module):
    def __init__(self):
        super(SDVAE_EMA, self).__init__()
        self.model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", torch_dtype=torch.bfloat16)
        self.model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
    def encode(self, x):
        return self.model.encode(x).latents
    
    def decode(self, x):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                x = self.model.decode(x / self.model.config.scaling_factor).sample
        x = (x + 1) / 2
        x = x.clamp(0, 1)
        x = (x * 255).to(torch.uint8)
        x = x.permute(0, 2, 3, 1)
        return x
