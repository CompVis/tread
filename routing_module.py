import torch

class Router:
    def __init__(self, seed=42):
        self.seed = seed
        
    def get_mask(self, x, mask_ratio=0.0, l1_reg=0.0, inverse=False):
        batch_size, num_patches, _ = x.shape
        device = x.device
        num_mask = int(num_patches * mask_ratio)
        num_keep = num_patches - num_mask
        token_magnitudes = x.abs().sum(dim=-1)
        min_mags = token_magnitudes.min(dim=1, keepdim=True)[0]
        max_mags = token_magnitudes.max(dim=1, keepdim=True)[0]
        token_magnitudes = (token_magnitudes - min_mags) / (max_mags - min_mags + 1e-8)
        if inverse:
            adjusted_magnitudes = 1.0 - token_magnitudes
        else:
            adjusted_magnitudes = token_magnitudes
        noise_random = torch.rand(batch_size, num_patches, device=device)
        noise = (1.0 - l1_reg) * noise_random + l1_reg * adjusted_magnitudes
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :num_keep]
        ids_mask = ids_shuffle[:, num_keep:]
        mask = torch.ones((batch_size, num_patches), device=device, dtype=torch.bool)
        mask.scatter_(1, ids_keep, False)
        return {
            'mask': mask,           
            'ids_keep': ids_keep,
            'ids_mask': ids_mask,
            'ids_shuffle': ids_shuffle,
            'ids_restore': ids_restore
        }
    
    def start_route(self, x, mask_info):
        ids_shuffle = mask_info['ids_shuffle']
        num_keep = mask_info['ids_keep'].size(1)
        batch_indices = torch.arange(x.size(0), device=x.device).unsqueeze(-1)
        x_shuffled = x.gather(1, ids_shuffle.unsqueeze(-1).expand(-1, -1, x.size(2)))
        masked_x = x_shuffled[:, :num_keep, :]
        return masked_x
    
    def end_route(self, masked_x, mask_info, original_x=None, mask_token=0.0):
        batch_size, num_patches = mask_info['mask'].shape
        num_keep = masked_x.size(1)
        dim = masked_x.size(2)
        device = masked_x.device
        ids_restore = mask_info['ids_restore']
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(-1)
        x_unshuffled = torch.empty((batch_size, num_patches, dim), device=device)
        x_unshuffled[:, :num_keep, :] = masked_x
        if original_x is not None:
            x_shuffled = original_x.gather(1, mask_info['ids_shuffle'].unsqueeze(-1).expand(-1, -1, dim))
            x_unshuffled[:, num_keep:, :] = x_shuffled[:, num_keep:, :]
        else:
            x_unshuffled[:, num_keep:, :].fill_(mask_token)
        x_unmasked = x_unshuffled.gather(1, ids_restore.unsqueeze(-1).expand(-1, -1, dim))
        return x_unmasked