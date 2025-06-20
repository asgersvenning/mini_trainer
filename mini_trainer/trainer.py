import datetime
import os
import time
import warnings
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from torch import autocast
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from mini_trainer import TQDM
from mini_trainer.utils import (TERMINAL_WIDTH, reduce_across_processes,
                                save_on_master)
from mini_trainer.utils.logging import MultiLogger


def train_one_epoch(
        model : nn.Module, 
        criterion : _Loss, 
        optimizer : Optimizer, 
        scaler : GradScaler,
        lr_scheduler : LRScheduler,
        data_loader : DataLoader, 
        epoch : int, 
        logger : MultiLogger,
        preprocess : Callable=lambda x : x,
        augmentation : Callable=lambda x : x,
        regularizer : Callable[[nn.Module], torch.Tensor]=lambda _: 0.,
        clip_grad_norm : Optional[float]=1,
        device : torch.types.Device=torch.device("cpu"),
        dtype : torch.dtype=torch.float32,
    ):
    model.train()
    pbar = TQDM(data_loader, total=len(data_loader), ncols=TERMINAL_WIDTH, leave=False)
    logger.update(epoch=epoch, type="train")

    nan_errs = 0

    start_time = time.time()
    for i, (batch, target) in enumerate(pbar):
        batch, target = batch.to(device), target.to(device)
        if len(batch.shape) == 3:
                batch = batch.unsqueeze(0)
        with autocast(device_type=device.type, dtype=dtype):
            output = model(preprocess(augmentation(batch)))
            loss : Union[list[torch.Tensor], torch.Tensor] = criterion(output, target) 
        if isinstance(loss, torch.Tensor) and loss.numel() == 1:
            loss = [loss]
        optimizer.zero_grad()
        if not all([torch.isfinite(term).all() for term in loss]):
            nan_errs += 1
            if nan_errs < 5:
                continue
            else:
                raise RuntimeError(f'Interrupted training due to persistent nan\'s detected in the loss.')
        else:
            nan_errs = 0
        
        scaler.scale(sum(loss) + regularizer(model)).backward()
        scaler.unscale_(optimizer)
        if clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        
        logger.consume(
            index=i,
            batch=batch, 
            target=target, 
            prediction=output, 
            loss=loss, 
            optimizer=optimizer, 
            start_time=start_time
        )
        pbar.set_description_str(logger.status(), i % 25 == 0)
        start_time = time.time()

    model.eval()

def evaluate(
        model : nn.Module, 
        criterion : _Loss, 
        data_loader : DataLoader, 
        epoch : int,
        logger : MultiLogger,
        preprocess : Callable=lambda x : x,
        device : torch.types.Device=torch.device("cpu"),
        dtype : torch.dtype=torch.float32
    ):
    model.eval()
    pbar = TQDM(data_loader, desc="Evaluation", total=len(data_loader), ncols=TERMINAL_WIDTH, leave=False)
    logger.update(epoch=epoch, type="eval")

    num_processed_samples = 0
    start_time = time.time()
    for i, (batch, target) in enumerate(pbar):
        with torch.inference_mode():
            batch, target = batch.to(device), target.to(device)
            if len(batch.shape) == 3:
                    batch = batch.unsqueeze(0)
            with autocast(device_type=device.type, dtype=dtype):
                output = model(preprocess(batch))
                loss = criterion(output, target)
            logger.consume(
                index=i, 
                batch=batch, 
                target=target, 
                prediction=output, 
                loss=loss, 
                optimizer=None, 
                start_time=start_time
            )
        pbar.set_description_str(logger.status(), i % 25 == 0)
        num_processed_samples += len(batch)
        start_time = time.time()
    
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

    if logger.verbose:
        print(logger.summary())
    logger.figures(model)
    return logger.statistics_storage[logger.canonical_statistic][-1]

def train(
        model : nn.Module, 
        train_loader : DataLoader, 
        val_loader : DataLoader,
        criterion : _Loss, 
        optimizer : Optimizer, 
        lr_scheduler : LRScheduler,
        logger : MultiLogger,
        epochs : int, 
        start_epoch : int = 0,
        preprocess : Callable=lambda x : x,
        augmentation : Callable=lambda x : x,
        regularizer : Callable[[nn.Module], torch.Tensor]=lambda _: 0.,
        device : torch.types.Device=torch.device("cpu"),
        dtype : torch.dtype=torch.float32,
        output_dir : Optional[str]=None,
        weight_store_rate : Optional[int]=None,
        **kwargs
    ):
    # model = torch.compile(model)
    if logger.verbose:
        print("Start training")
    start_time = time.time()

    scaler = GradScaler(device=device)
    for epoch in range(start_epoch, epochs):
        train_one_epoch(model, criterion, optimizer, scaler, lr_scheduler, train_loader, epoch, logger, preprocess, augmentation, regularizer, device=device, dtype=dtype, **kwargs)
        evaluate(model, criterion, val_loader, epoch, logger, preprocess, device=device, dtype=dtype)
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
            if weight_store_rate is not None and epoch % weight_store_rate == 0:
                save_on_master(checkpoint, os.path.join(output_dir, f"model_{epoch}.pth"))
            save_on_master(checkpoint, os.path.join(output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if logger.verbose:
        print(f"Training time {total_time_str}")