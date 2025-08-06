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
        x_shuffled = x.gather(1, ids_shuffle.unsqueeze(-1).expand(-1, -1, x.size(2)))
        masked_x = x_shuffled[:, :num_keep, :]
        return masked_x
    
    def end_route(self, masked_x, mask_info, original_x):
        batch_size, num_patches = mask_info['mask'].shape
        num_keep = masked_x.size(1)
        dim = masked_x.size(2)
        device = masked_x.device
        ids_restore = mask_info['ids_restore']
        x_unshuffled = torch.empty((batch_size, num_patches, dim), device=device)
        x_unshuffled[:, :num_keep, :] = masked_x
        x_shuffled = original_x.gather(1, mask_info['ids_shuffle'].unsqueeze(-1).expand(-1, -1, dim))
        x_unshuffled[:, num_keep:, :] = x_shuffled[:, num_keep:, :]
        x_unmasked = x_unshuffled.gather(1, ids_restore.unsqueeze(-1).expand(-1, -1, dim))
        return x_unmasked