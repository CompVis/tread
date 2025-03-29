# This code is adapted from https://github.com/NVlabs/edm/blob/main/fid.py and https://github.com/Anima-Lab/MaskDiT/blob/master/fid.py
# The original code is licensed under a Creative Commons 
# Attribution-NonCommercial-ShareAlike 4.0 International License, which is can be found at licenses/LICENSE_EDM.txt. 

import argparse
from multiprocessing import Process

import click
import tqdm
import pickle
import numpy as np
import scipy.linalg
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import sys
import os
    
from utils.train_helper import *
from datasets import ImageFolderDataset
from torch_utils import * 


#----------------------------------------------------------------------------

def calculate_inception_stats(
        image_path, num_expected=None, seed=0, max_batch_size=64,
        num_workers=3, prefetch_factor=2, device=torch.device('cuda'), inception_path=None
):
    num_gpus = torch.cuda.device_count()
    use_distributed = num_gpus > 1
    if use_distributed:
        if dist.get_rank() != 0:
            dist.barrier()

    detector_path = inception_path

    print('Loading Inception-v3 model from disk...')
    with open(detector_path, 'rb') as f:
        detector_net = pickle.load(f).to(device)

    detector_kwargs = dict(return_features=True)
    feature_dim = 2048

    dataset_obj = ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)
    if num_expected is not None and len(dataset_obj) < num_expected:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_expected}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    if use_distributed:
        if dist.get_rank() == 0:
            dist.barrier()

    if use_distributed:
        num_batches = ((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    else:
        num_batches = (len(dataset_obj) - 1) // max_batch_size + 1
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    if use_distributed:
        rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    else:
        rank_batches = all_batches
    data_loader = DataLoader(dataset_obj, batch_sampler=rank_batches, num_workers=num_workers, prefetch_factor=prefetch_factor)

    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    if use_distributed:
        for images, _labels in tqdm.tqdm(data_loader, unit='batch', disable=(dist.get_rank() != 0)):
            if use_distributed:
                dist.barrier()
            if images.shape[0] == 0:
                continue
            if images.shape[1] == 1:
                images = images.repeat([1, 3, 1, 1])
            features = detector_net(images.to(device), **detector_kwargs).to(torch.float64)
            mu += features.sum(0)
            sigma += features.T @ features
    else:
        for images, _labels in tqdm.tqdm(data_loader, unit='batch'):
            if images.shape[0] == 0:
                continue
            if images.shape[1] == 1:
                images = images.repeat([1, 3, 1, 1])
            features = detector_net(images.to(device), **detector_kwargs).to(torch.float64)
            mu += features.sum(0)
            sigma += features.T @ features

    if use_distributed:
        dist.all_reduce(mu)
        dist.all_reduce(sigma)
    mu /= len(dataset_obj)
    sigma -= mu.ger(mu) * len(dataset_obj)
    sigma /= len(dataset_obj) - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()


def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))



def calc(image_path, ref_path, num_expected, seed, batch, inception_path):
    """Calculate FID for a given set of images."""
    num_gpus = torch.cuda.device_count()
    use_distributed = num_gpus > 1

    ref = None
    if use_distributed:
        if dist.get_rank() == 0:
            assert ref_path.endswith('.npz')
            ref = dict(np.load(ref_path))
    else:
        assert ref_path.endswith('.npz')
        ref = dict(np.load(ref_path))

    mu, sigma = calculate_inception_stats(image_path=image_path, num_expected=num_expected, seed=seed, max_batch_size=batch, inception_path=inception_path)
    # mprint('Calculating FID...')
    fid = None
    if use_distributed:
        if dist.get_rank() == 0:
            fid = calculate_fid_from_inception_stats(mu, sigma, ref['mu'], ref['sigma'])
            print(f'{fid:g}')
            return fid

    else:
        fid = calculate_fid_from_inception_stats(mu, sigma, ref['mu'], ref['sigma'])
        print(f'{fid:g}')
        return fid

def ref(dataset_path, dest_path, batch):
    """Calculate dataset reference statistics needed by 'calc'."""
    num_gpus = torch.cuda.device_count()
    use_distributed = num_gpus > 1
    mu, sigma = calculate_inception_stats(image_path=dataset_path, max_batch_size=batch)
    if use_distributed:
        if dist.get_rank() == 0:
            if os.path.dirname(dest_path):
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            np.savez(dest_path, mu=mu, sigma=sigma)
    else:
        if os.path.dirname(dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        np.savez(dest_path, mu=mu, sigma=sigma)
    if use_distributed:
        dist.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('fid parameters')

    # ddp
    parser.add_argument('--num_proc_node', type=int, default=1, help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1, help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0, help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0, help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='localhost', help='address for master')

    # fid
    parser.add_argument('--mode', type=str, required=True, choices=['calc', 'ref'], help='Calcalute FID or store reference statistics')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the images')
    parser.add_argument('--ref_path', type=str, default='path_to_reference_batch', help='Dataset reference statistics')
    parser.add_argument('--num_expected', type=int, default=50000, help='Number of images to use')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for selecting the images')
    parser.add_argument('--batch', type=int, default=64, help='Maximum batch size per GPU')
    parser.add_argument('--inception_path', type=str, default='path_to_network', help='Path to the inception model')

    args = parser.parse_args()
    args.global_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    func = lambda args: calc(args.image_path, args.ref_path, args.num_expected, args.seed, args.batch) \
        if args.mode == 'calc' else lambda args: ref(args.image_path, args.ref_path, args.batch)

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            args.global_rank = rank + args.node_rank * args.num_process_per_node
            p = Process(target=init_processes, args=(func, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        print('Single GPU run')
        assert args.global_size == 1 and args.local_rank == 0
        args.global_rank = 0
        init_processes(func, args)