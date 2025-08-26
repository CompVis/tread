import torch

class Router:
    def __init__(self, seed=42):
        self.seed = seed
        
    def get_mask(self, x, selection_rate=0.0):
        batch_size, num_patches, _ = x.shape
        device = x.device
        num_mask = int(num_patches * selection_rate)
        num_keep = num_patches - num_mask
        noise_random = torch.rand(batch_size, num_patches, device=device)
        ids_shuffle = torch.argsort(noise_random, dim=1)
        ids_keep = ids_shuffle[:, :num_keep]
        return ids_keep
    
    def start_route(self, x, ids_keep):
        x_masked = x.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, x.size(2)))
        return x_masked
    
    def end_route(self, masked_x, ids_keep, original_x):
        # (jerry) scatter is out-of-place, so this is safe
        x_unmasked = original_x.scatter(
            1, ids_keep.unsqueeze(-1).expand(-1, -1, original_x.size(2)), masked_x
        )
        return x_unmasked
