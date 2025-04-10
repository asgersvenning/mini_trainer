import datetime
import os
import time
import warnings
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import autocast
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm as TQDM

from utils import (MetricLogger, SmoothedValue, accuracy,
                   reduce_across_processes, save_on_master)


def train_one_epoch(
        model : nn.Module, 
        criterion : _Loss, 
        optimizer : Optimizer, 
        lr_scheduler : LRScheduler,
        data_loader : DataLoader, 
        epoch : int, 
        preprocess : Callable=lambda x : x,
        augmentation : Callable=lambda x : x,
        clip_grad_norm : Optional[float]=1,
        device : torch.types.Device=torch.device("cpu"),
        dtype : torch.dtype=torch.float32
    ):
    model.train()
    
    pbar = TQDM(data_loader, total=len(data_loader), leave=False)
    # pbar = data_loader
    
    metric_logger = MetricLogger(delimiter="  ", printer=lambda x : pbar.set_description_str(x))
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"

    for i, (batch, target) in enumerate(metric_logger.log_every(pbar, 25, header)):
        start_time = time.time()
        with autocast(device_type=device.type, dtype=dtype):
            if len(batch.shape) == 3:
                batch = batch.unsqueeze(0)
            output = model(preprocess(augmentation(batch)))
            loss = criterion(output, target) 

        optimizer.zero_grad()
        loss.backward()
        if clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        lr_scheduler.step()

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = batch.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
    
    model.eval()

def evaluate(
        model, 
        criterion, 
        data_loader, 
        preprocess : Callable=lambda x : x,
        print_freq=100, 
        log_suffix="",
        device : torch.types.Device=torch.device("cpu"),
        dtype : torch.dtype=torch.float32
    ):
    model.eval()
    header = f"Test: {log_suffix}"
    pbar = TQDM(data_loader, desc="Evaluation", total=len(data_loader), leave=False)
    metric_logger = MetricLogger(delimiter="  ", printer = lambda x : pbar.set_description_str(x))

    num_processed_samples = 0
    with torch.inference_mode():
        for batch, target in metric_logger.log_every(pbar, print_freq, header):
            if len(batch.shape) == 3:
                batch = batch.unsqueeze(0)
            with autocast(device_type=device.type, dtype=dtype):
                output = model(preprocess(batch))
                loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = batch.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes
    num_processed_samples = reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg

def train(
        model : nn.Module, 
        train_loader : DataLoader, 
        test_loader : DataLoader,
        criterion : _Loss, 
        optimizer : Optimizer, 
        lr_scheduler : LRScheduler,
        epochs : int, 
        start_epoch : int = 0,
        preprocess : Callable=lambda x : x,
        augmentation : Callable=lambda x : x,
        device : torch.types.Device=torch.device("cpu"),
        dtype : torch.dtype=torch.float32,
        output_dir : Optional[str]=None,
        **kwargs
    ):
    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        train_one_epoch(model, criterion, optimizer, lr_scheduler, train_loader, epoch, preprocess, augmentation, device=device, dtype=dtype, **kwargs)
        evaluate(model, criterion, test_loader, preprocess, device=device, dtype=dtype)
        if output_dir:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
            }
            # if model_ema:
            #     checkpoint["model_ema"] = model_ema.state_dict()
            # if scaler:
            #     checkpoint["scaler"] = scaler.state_dict()
            save_on_master(checkpoint, os.path.join(output_dir, f"model_{epoch}.pth"))
            save_on_master(checkpoint, os.path.join(output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")