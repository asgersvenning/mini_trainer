import datetime
import os
import time
import warnings
from collections.abc import Callable

import torch
import torch.nn as nn
from torch import autocast
from torch.amp import GradScaler
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from mini_trainer import TQDM
from mini_trainer.utils import (TERMINAL_WIDTH, is_dist_avail_and_initialized,
                                reduce_across_processes, save_on_master)
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
        clip_grad_norm : float | None=1,
        device : torch.types.Device=torch.device("cpu"),
        dtype : torch.dtype=torch.float32,
    ):
    """
    Run one training epoch.

    Args:
        model: Model under training.
        criterion: Loss function; may return a scalar tensor or a list of tensors.
        optimizer: Optimizer used for parameter updates.
        scaler: AMP gradient scaler.
        lr_scheduler: Learning rate scheduler stepped per batch.
        data_loader: Dataloader yielding mini-batches of ``(inputs, targets)``.
        epoch: Zero-based epoch index.
        logger: Multi-backend logger used to record metrics and figures.
        preprocess: Function applied to tensors before passing to the model.
        augmentation: Training-time augmentation applied before preprocess.
        regularizer: Callable that returns an extra scalar loss term from the model.
        clip_grad_norm: Max gradient norm; disabled if ``None``.
        device: Target device for training (e.g., ``cuda:0``).
        dtype: AMP/autocast data type for forward pass.

    Raises:
        RuntimeError: If non-finite loss persists across several steps or input shape is invalid.
    """
    model.train()
    pbar = TQDM(data_loader, total=len(data_loader), ncols=TERMINAL_WIDTH, leave=False)
    logger.update(epoch=epoch, type="train")

    nan_errs = 0

    start_time = time.time()
    for i, (batch, target) in enumerate(pbar):
        batch, target = batch.to(device), target.to(device)
        if len(batch.shape) != 4:
            raise RuntimeError(f'Incorrect {batch.shape=}, expected 4 dimensions, not {len(batch.shape)}.')
        with autocast(device_type=device.type, dtype=dtype):
            output = model(preprocess(augmentation(batch)))
            loss : list[torch.Tensor] | torch.Tensor = criterion(output, target) 
        if isinstance(loss, torch.Tensor) and loss.numel() == 1:
            loss = [loss]
        optimizer.zero_grad()
        if not all([torch.isfinite(term).all() for term in loss]):
            nan_errs += 1
            if nan_errs < 5:
                continue
            else:
                raise RuntimeError('Interrupted training due to persistent nan\'s detected in the loss.')
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
    """
    Evaluate the model for one validation epoch.

    Args:
        model: Model in evaluation mode.
        criterion: Loss function compatible with the model outputs/targets.
        data_loader: Validation dataloader yielding mini-batches.
        epoch: Zero-based epoch index.
        logger: Logger used to record metrics and figures.
        preprocess: Preprocess function applied to tensors before inference.
        device: Target device for evaluation.
        dtype: AMP/autocast data type for inference.

    Returns:
        The most recent value of the canonical statistic recorded by the logger.
    """
    model.eval()
    pbar = TQDM(data_loader, desc="Evaluation", total=len(data_loader), ncols=TERMINAL_WIDTH, leave=False)
    logger.update(epoch=epoch, type="eval")

    num_processed_samples = 0
    start_time = time.time()
    for i, (batch, target) in enumerate(pbar):
        with torch.inference_mode():
            batch, target = batch.to(device), target.to(device)
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
        and (not is_dist_avail_and_initialized() or torch.distributed.get_rank() == 0)
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
        output_dir : str | None=None,
        weight_store_rate : int | None=None,
        **kwargs
    ):
    """
    Full training loop across epochs with periodic evaluation and checkpointing.

    Args:
        model: Model to train.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        criterion: Loss function.
        optimizer: Optimizer instance.
        lr_scheduler: LR scheduler stepped every training batch.
        logger: Logger used for metrics, summaries and figures.
        epochs: Total number of epochs to run.
        start_epoch: Initial epoch index when resuming from a checkpoint.
        preprocess: Preprocess function applied prior to the model.
        augmentation: Augmentation function used during training only.
        regularizer: Callable returning an extra scalar loss term from the model.
        device: Target device.
        dtype: AMP/autocast data type for forward/eval passes.
        output_dir: If provided, checkpoints are written here.
        weight_store_rate: Store a snapshot every ``weight_store_rate`` epochs if set.
        **kwargs: Forwarded to lower-level helpers.
    """
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
