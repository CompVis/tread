import sys
import os
import warnings
import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator, DistributedDataParallelKwargs
from utils.train_helper import update_ema, requires_grad
from copy import deepcopy
from time import time
import torch.distributed as dist
from fid import calc
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import glob
import numpy as np
from itertools import islice
import webdataset as wds
import pickle
import io

warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")


# WebDataset Helper Function
def nodesplitter(src, group=None):
    rank, world_size, worker, num_workers = wds.utils.pytorch_worker_info()
    if world_size > 1:
        for s in islice(src, rank, None, world_size):
            yield s
    else:
        for s in src:
            yield s

def get_file_paths(directory, mode):
    return [os.path.join(directory, file) for file in os.listdir(directory) if mode in file]

def split_by_proc(data_list, global_rank, total_size):
    '''
    Evenly split the data_list into total_size parts and return the part indexed by global_rank.
    '''
    assert len(data_list) >= total_size
    assert global_rank < total_size
    return data_list[global_rank::total_size]

def decode_data(item):
    output = {}
    
    # Decode the latent numpy array
    latent_array = np.load(io.BytesIO(item["latent.npy"]))
    latent = torch.from_numpy(latent_array)
    output['latent'] = latent
    
    # Decode the class label
    label = int(item['cls'].decode('utf-8'))
    output['label'] = label        
    return output

def make_loader(root, mode='ImageNetTrain-train', batch_size=32, 
                num_workers=4, cache_dir=None, 
                resampled=False, world_size=1, total_num=1281167, 
                bufsize=1000, initial=100):
    data_list = get_file_paths(root, mode)
    print("Number of shards:", len(data_list))
    num_batches_in_total = total_num // (batch_size * world_size)
    
    if resampled:
        repeat = True
        splitter = nodesplitter
    else:
        repeat = False
        splitter = nodesplitter

    # Create the WebDataset pipeline and pass the flag to decode_data via a lambda
    dataset = (
        wds.WebDataset(
            data_list, 
            cache_dir=cache_dir,
            repeat=repeat,
            resampled=resampled, 
            handler=wds.handlers.warn_and_stop, 
            nodesplitter=splitter,
            shardshuffle=True
        )
        .shuffle(bufsize, initial=initial)
        .map(lambda item: decode_data(item), handler=wds.handlers.warn_and_stop)
    )
    

    dataset = dataset.to_tuple('latent', 'label')
    
    dataset = dataset.batched(batch_size, partial=False)
    
    loader = wds.WebLoader(
        dataset, 
        batch_size=None, 
        num_workers=num_workers, 
        shuffle=False, 
        persistent_workers=True
    )
    
    if resampled:
        loader = loader.with_epoch(num_batches_in_total)
        
    return loader
def rzprint(*args, **kwargs):
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)

@hydra.main(config_path="configs", config_name="config")
def train(cfg: DictConfig):
    
    print(OmegaConf.to_yaml(cfg))

    data_config = cfg.dataset
    model_config = cfg.model
    
    experiment_dir = os.path.join(cfg.results_dir, cfg.run_name)

    ##############################################################
    # INIT
    ##############################################################
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    size = accelerator.num_processes
    rank = accelerator.process_index
    rzprint("Init Accelerator.")
    
    model = hydra.utils.instantiate(model_config.model).to(device)
    optimizer = hydra.utils.instantiate(cfg.train.optimizer, model.parameters())
    rzprint("Init model and optimizer.")
    
    diffuser = hydra.utils.instantiate(cfg.train.diffuser)
    model = diffuser.wrap_model_with_precond(model)
    ema = deepcopy(model)
    requires_grad(ema, False) 
       
    if cfg.load_ckpt:
        load_path = os.path.join(experiment_dir, 'latest.pt')
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            ema.load_state_dict(checkpoint['ema_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            step = checkpoint['step']
            rzprint(f"Loaded checkpoint from {load_path} at step {step}")
        else:
            rzprint(f"No checkpoint found at {load_path}")
            step = 0
    else:
        step = 0
        
    model, ema, optimizer = accelerator.prepare(model, ema, optimizer)
    model.train()
    ema.eval()
    rzprint("Init diffuser.")
    
    ##############################################################
    # DATA
    ##############################################################
    total_batch_size = cfg.train.general.batch_size
    batch_size_per_device = total_batch_size // size
    rzprint(f"Batch size per device: {batch_size_per_device}")
    rzprint(f"Total batch size: {total_batch_size}")
    loader = make_loader(
        cfg.dataset.train_path,
        mode='train',
        batch_size=batch_size_per_device,
        num_workers=data_config.num_workers,
        resampled=False,
        total_num=data_config.total_num
    )
    rzprint("Init data loader.")
    
    if cfg.log_wandb and rank == 0:
        wandb.init(project=cfg.wandb.project, config=OmegaConf.to_container(cfg))
        
    running_loss = 0
    start_time = time()

    rzprint("Starting training loop...")
    for epoch in range(cfg.train.general.max_epochs):
        rzprint(f"Epoch {epoch + 1}/{cfg.train.general.max_epochs}")
        for x, cond in loader:
            ##############################################################
            # TRAIN STEP
            ##############################################################
            x = x.to(accelerator.device)
            cond = cond.to(accelerator.device)

            with accelerator.autocast():
                loss = diffuser.get_training_loss(
                    model, 
                    x,
                    cond.to(torch.long),
                    class_drop_prob=cfg.train.general.class_drop_prob,
                )
                loss = loss.mean()

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            
            update_ema(ema, model.module)
            
            running_loss += loss.item()
            
            ##############################################################
            # LOGGING
            ##############################################################
            if step % cfg.train.logging.log_interval == 0 and step > 0:
                elapsed = time() - start_time
                steps_per_sec = cfg.train.logging.log_interval / elapsed
                avg_loss = running_loss / cfg.train.logging.log_interval
                rzprint(f"Step {step}: Loss: {avg_loss:.4f}, Steps/sec: {steps_per_sec:.2f} \n")
                if cfg.log_wandb and rank == 0:
                    wandb.log({"loss": avg_loss, "steps_per_sec": steps_per_sec}, step=step)
                running_loss = 0
                start_time = time()
                
            ##############################################################
            # EVAL
            ##############################################################
            if step % cfg.train.eval.eval_interval == 0 and cfg.enable_eval and step > 0:
                for cfg_scale in cfg.train.eval.cfg_scales:
                    cfg.train.eval.cfg_scale = cfg_scale
                    
                    outdir = os.path.join(experiment_dir, 'fid')
                    os.makedirs(outdir, exist_ok=True)
                    rzprint(f"FID Folder: {outdir}")
                    rzprint(f"EMA device: {next(ema.parameters()).device}")
                    start_time = time()
                    diffuser.generate(cfg.train.eval, ema, device, rank, size, outdir=outdir)
                    accelerator.wait_for_everyone()
                    elapsed = time() - start_time
                    rzprint(f"Time taken to generate samples: {elapsed:.2f}s")
                    fid = calc(outdir, data_config.ref_path, cfg.train.eval.fid_num_samples, cfg.global_seed, cfg.train.eval.fid_batch_size, cfg.train.eval.inception_path)
                    accelerator.wait_for_everyone()
                    cfg.train.eval.cfg_scale = None
                    if rank == 0:
                        rzprint(f"FID (CFG:{cfg_scale}): {fid}")
                        if cfg.log_wandb:
                            wandb.log({f"FID (CFG:{cfg_scale})": fid}, step=step)

                            num_samples = 16
                            image_files = sorted(glob.glob(os.path.join(outdir, '*.png')))
                            image_list = []
                            for img_file in image_files[:num_samples]:
                                img = Image.open(img_file).convert('RGB')
                                transform = transforms.ToTensor()
                                img_tensor = transform(img)
                                image_list.append(img_tensor)

                            if len(image_list) > 0:
                                grid = vutils.make_grid(image_list, nrow=int(np.sqrt(num_samples)), normalize=True)
                                wandb.log({f"FID (CFG:{cfg_scale}) Samples": [wandb.Image(grid, caption="Generated Samples")]}, step=step)

            ##############################################################
            # CHECKPOINT
            ##############################################################
            if cfg.save_ckpt and step % cfg.train.eval.eval_interval == 0 and step > 0:
                save_path = os.path.join(experiment_dir, f'step_{step:06d}.pt')
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model.module)
                unwrapped_ema = accelerator.unwrap_model(ema)
                if accelerator.is_main_process:
                    checkpoint = {
                        'model_state_dict': unwrapped_model.state_dict(),
                        'ema_state_dict': unwrapped_ema.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'step': step,
                    }
                    torch.save(checkpoint, save_path)
                    # Save latest checkpoint
                    latest_path = os.path.join(experiment_dir, 'latest.pt')
                    torch.save(checkpoint, latest_path)
                    
            step += 1    
                
                    
if __name__ == '__main__':
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    train()
