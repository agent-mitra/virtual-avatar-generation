"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataset.video_dataset import VideoDataset
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse, sys, re
import logging
import os
import torchvision.transforms as T
import numpy as np
import wandb
from PIL import Image
from diffusion_transformer import DiT_action
from diffusion import create_diffusion 
from torch.cuda.amp import autocast, GradScaler
from torch.nn import functional as F
from torchvision.io import read_image
import sys
sys.path.append('..')
from backbones import create_backbone
HOME_DIR = os.path.join(os.environ.get("HOME"), "virtual-avatar-generation")

#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def extract_number(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0])

def getFrameTensors(path, image_files, newH, newW, device):
    transform = torch.nn.Sequential(
        T.Resize((newH, newW), interpolation=T.InterpolationMode.BICUBIC),
        T.ConvertImageDtype(torch.float),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    )
    imgs_tensor_orig = torch.zeros(len(image_files), 3, newH, newW, device=device)
    for i, image_file in enumerate(image_files):
        img = read_image(os.path.join(path, image_file)).to(device)
        imgs_tensor_orig[i] = transform(img)
    return imgs_tensor_orig

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def custom_collate_fn(batch):
    # Unzip the batch into separate lists for x and y
    x, y = zip(*batch)
    
    # Stack x to create a single tensor for the batch
    x = torch.stack(x, dim=0)
    
    # y is already a list of values, so you can directly return it
    
    return x, list(y)

import gc
#################################################################################
#                                  Training Loop                                #
#################################################################################
import datetime
def main(args):
    """
    Trains a new DiT model for climbing.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    # Setup DDP:
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=5400))
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    print(f"Global batch size is {args.global_batch_size} with world size {dist.get_world_size()}")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = "DiT-Climb"  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        my_id = "<ID>" #TODO: if you want to re-start from a wandb run, uncomment id and resume below in wandb.init as well
        wandb.init(
            # Set the project where this run will be logged
            project="climb-v1", 
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"DiT-Climb-L-v1", 
            # id=my_id,
            # resume="allow",
            # Track hyperparameters and run metadata
            config={
            "learning_rate": 1e-4,
            "architecture": "Diffusion transformer",
            "epochs": args.epochs,
            })
    else:
        logger = create_logger(None)

    # Create model:
    model = DiT_action()
    # Note that parameter initialization is done within the DiT constructor
    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="", predict_xstart=True)  # default: 1k steps, linear noise schedule, predict_xstart=True can be true
    if rank == 0:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0) #change lr and weight_decay
    
    start_epoch = 0
    checkpoint_path = args.ckpt_path
    if checkpoint_path is not None:
        if rank == 0:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        model.module.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        opt.load_state_dict(checkpoint["opt"])
        start_epoch = checkpoint["epoch"] #next one started at 19 so.
        if rank == 0:
            logger.info(f"Checkpoint loaded successfully")
        train_steps = checkpoint["steps"]
        log_steps = checkpoint["steps"]

    # Setup data:
    dataset = VideoDataset() #load video dataset here
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    
    
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn,
    )

    if rank == 0:
        logger.info(f"Dataset loaded!")

    # Prepare models for training:
    if checkpoint_path is None: #no need
        print("initialized ema!")
        update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
        train_steps = 0
        log_steps = 0
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    # Variables for monitoring/logging purposes:
    running_loss = 0
    start_time = time()
    scaler = GradScaler()
    if rank == 0:
        logger.info(f"Training for {args.epochs} epochs...")
        wandb.watch(model, log="all", log_freq=100)
    backbone = create_backbone(model="large", device=device)
    zero_vector = torch.from_numpy(torch.load(f"{HOME_DIR}/model/backbones/dinov2_zero_vector.pt", map_location=device)).to(device)
    seqLen = 1100
    for epoch in range(start_epoch, args.epochs): 
        sampler.set_epoch(epoch)
        if rank == 0:
            logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y_arr = []
            for tuplex in y:
                inpaintedFrames_path, contextFrames_path, (newH, newW), context_image_files, inpainted_image_files = tuplex
                context_tensor = getFrameTensors(contextFrames_path, context_image_files, newH, newW, device)
                video_tensor = getFrameTensors(inpaintedFrames_path, inpainted_image_files, newH, newW, device)
                with torch.no_grad():
                    context_feature_vector = backbone.forward_features(context_tensor.to(device))['x_norm_patchtokens']    
                vid_vecs = []
                with torch.no_grad():
                    for batch in range(0, video_tensor.shape[0], 350):
                        video_feature_vector = backbone.forward_features(video_tensor[batch: batch + 350].to(device))['x_norm_patchtokens']
                        vid_vecs.append(video_feature_vector)
                        del video_feature_vector
                video_feature_vectors = torch.cat(vid_vecs, dim=0).to(device)
                assert video_feature_vectors.shape[0] == video_tensor.shape[0]
                combined_vector = torch.cat((context_feature_vector, video_feature_vectors), dim=0)[:seqLen].to(device)
                if combined_vector.shape[0] < seqLen:
                    toAdd = seqLen - combined_vector.shape[0]
                    zeros_to_add = zero_vector.repeat(toAdd, 1, 1)
                    combined_vector = torch.cat((combined_vector, zeros_to_add), dim=0)
                    del zeros_to_add
                assert combined_vector.shape[0] == seqLen
                y_arr.append(combined_vector) 
                del combined_vector, context_feature_vector, video_feature_vectors, vid_vecs
                torch.cuda.empty_cache()
                gc.collect()
            y = torch.stack(y_arr, dim=0).to(device)
            del y_arr
            if rank == 0:
                logging.info(f"Data collated, running through model {epoch}")
            
            #x is the sequences of motion vectors, y is the video frames and context frames
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            
            with autocast():
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
            if rank == 0:
                logging.info(f"One forward pass completed for epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            if rank == 0:
                logging.info(f"One backward pass completed for epoch {epoch}")
                
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                if rank == 0:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    wandb.log({"loss": avg_loss})
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "epoch": epoch,
                        "steps": train_steps,
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default="/home/shadeform/virtual-avatar-generation/CLIMB/CLIMB_diffusion/results")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10_000)
    args = parser.parse_args()
    main(args)