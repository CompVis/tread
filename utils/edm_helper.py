import time
import random
from typing import Dict, Union, Any, Optional

import numpy as np
import torch
import re
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from transformers import get_scheduler as _get_scheduler
from transformers.optimization import SchedulerType

def patchify(imgs, patch_size=2, num_channels=4):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p, c = patch_size, num_channels
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
    return x


def mae_loss(net, target, pred, mask, norm_pix_loss=True):
    target = patchify(target, net.model.patch_size, net.model.out_channels)
    pred = patchify(pred, net.model.patch_size, net.model.out_channels)
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum(dim=1) / mask.sum(dim=1)  # mean loss on removed patches, (N)
    assert loss.ndim == 1
    return loss

def mean_flat(tensor):
    return tensor.mean(dim=[1, 2, 3]) if tensor.ndim == 4 else tensor.mean()

def unwrap_model(model):
    # Placeholder for unwrapping distributed models
    return model

class TimeMeasurement:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.ema = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        if self.ema is None:
            self.ema = elapsed_time
        else:
            self.ema = self.alpha * elapsed_time + (1 - self.alpha) * self.ema

    def reset(self):
        self.ema = None


class NullObject:
    def __getattr__(self, name) -> "NullObject":
        return NullObject()

    def __call__(self, *args: Any, **kwds: Any) -> "NullObject":
        return NullObject()
    
    def __enter__(self) -> "NullObject":
        return self
    
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass


def set_seed(seed=42, cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def dict_to(d: Dict[str, Union[torch.Tensor, Any]], **to_kwargs) -> Dict[str, Union[torch.Tensor, Any]]:
    return {k: (v.to(**to_kwargs) if isinstance(v, torch.Tensor) else v) for k, v in d.items()}


# Taken from https://github.com/cloneofsimo/minRF/blob/main/advanced/main_t2i.py
# Thanks
# Apache 2.0 License
def _z3_params_to_fetch(param_list):
    return [p for p in param_list if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE]


def save_zero_three_model(model, global_rank, output_model_file, zero_stage=0):
    zero_stage_3 = zero_stage == 3

    model_to_save = model.module if hasattr(model, "module") else model
    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():
            if hasattr(v, "ds_id"):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v]), enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict

def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    scheduler_specific_kwargs: Optional[dict] = None,
):
    if name == "exponential_decay":

        def lr_lambda(current_step: int):
            return 0.5 ** ((current_step) / scheduler_specific_kwargs["t_decay"])

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        return _get_scheduler(
            name=name,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            scheduler_specific_kwargs=scheduler_specific_kwargs,
        )
    
class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

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

# Parse 'None' to None and others to float value
def parse_float_none(s):
    assert isinstance(s, str)
    return None if s == 'None' else float(s)

# Parse 'None' to None and others to str
def parse_str_none(s):
    assert isinstance(s, str)
    return None if s == 'None' else s

# Parse 'true' to True
def str2bool(s):
    return s.lower() in ['true', '1', 'yes']

def unwrap_model(model):
    """
    Unwrap a model from any distributed or compiled wrappers. 
    """
    if isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
        model = model._orig_mod
    if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
        model = model.module
    return model