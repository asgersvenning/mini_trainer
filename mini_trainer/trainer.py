import datetime
import logging
import os
import time
import warnings
from collections.abc import Callable
from typing import Literal

import torch
import torch.nn as nn
from torch import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.modules.loss import _Loss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from mini_trainer import TQDM
from mini_trainer.builders import EMATeacher
from mini_trainer.logging import MultiLogger
from mini_trainer.modeling import EmbeddingContext, SupervisionContext
from mini_trainer.utils import (
    TERMINAL_WIDTH,
    is_dist_avail_and_initialized,
    reduce_across_processes,
    save_on_master,
)

# from mini_trainer.contrastive import SupConLoss

# contrastive_criterion = SupConLoss(temperature=25, base_temperature=25)


log = logging.getLogger("mini_trainer")


def train_one_epoch(
    model: nn.Module,
    model_ema: EMATeacher,
    criterion: _Loss,
    optimizer: Optimizer,
    scaler: GradScaler,
    lr_scheduler: LRScheduler,
    data_loader: DataLoader,
    epoch: int,
    logger: MultiLogger,
    preprocess: Callable = lambda x: x,
    augmentation: Callable = lambda x: x,
    regularizer: Callable[[nn.Module], torch.Tensor | Literal[0]] = lambda x: 0,
    clip_grad_norm: float | None = 5,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
):
    """Run one training epoch.

    Args:
        model: Model under training.
        model_ema: Exponential Moving Average model (``mini_trainer.builders.EMATeacher``) linked to ``model``.
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
    log.debug(f"train_one_epoch starting for epoch {epoch}.")
    model.train()
    getattr(getattr(data_loader.batch_sampler, "sampler", lambda x: x), "set_epoch", lambda x: x)(epoch)

    n_batches = len(data_loader)
    pbar = TQDM(data_loader, total=n_batches, ncols=TERMINAL_WIDTH, leave=False)
    logger.update(epoch=epoch, type="train")
    logger.start_timing()

    nan_errs = 0
    distill_loss = 0

    start_time = time.time()
    for i, (batch, target) in enumerate(pbar):
        step = n_batches * epoch + i
        if len(batch.shape) != 4:
            raise RuntimeError(f"Incorrect {batch.shape=}, expected 4 dimensions, not {len(batch.shape)}.")
        batch, target = batch.to(device), target.to(device)
        with autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32), SupervisionContext(target), EmbeddingContext():
            logits = model(preprocess(augmentation(batch)))
            loss: list[torch.Tensor] | torch.Tensor = criterion(logits, target)
            # TODO: Add optional contrastive path
            # ctr_loss = contrastive_criterion()
            # If EMA is disabled ``distill_loss`` is ``0.0``
            distill_loss = model_ema.teach(step=step, input=preprocess(batch), student=logits)
            reg = regularizer(model)

        if isinstance(loss, torch.Tensor) and loss.numel() == 1:
            loss = [loss]
        optimizer.zero_grad()
        if not all([torch.isfinite(term).all() for term in loss]) or not torch.isfinite(torch.as_tensor(distill_loss)):
            nan_errs += 1
            if nan_errs < 5:
                continue
            else:
                raise RuntimeError("Interrupted training due to persistent nan's detected in the loss.")
        else:
            nan_errs = 0
        scaler.scale(sum(loss) + reg + distill_loss).backward()
        scaler.unscale_(optimizer)
        if clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        scaler.step(optimizer)
        # We need to check if the GradScaler has detected NaN gradients, which will
        # result in optimizer.step() being skipped, and the warning:
        # "UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`"
        # being thrown - even though these two calls are in the right order here.
        # See: https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930/7
        _scale = scaler.get_scale()
        scaler.update()
        _any_opt_stepped = _scale <= scaler.get_scale()
        if _any_opt_stepped:
            raw_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
            model_ema.update_parameters(step, raw_model)
            lr_scheduler.step()

        logger.consume(
            index=i,
            batch=batch,
            target=target,
            prediction=logits,
            loss=loss,
            optimizer=optimizer,
            start_time=start_time,
            distillation_loss=float(distill_loss.detach().item() if isinstance(distill_loss, torch.Tensor) else distill_loss),
            regularization=float(reg.detach().item() if isinstance(reg, torch.Tensor) else reg),
        )
        pbar.set_description_str(logger.status(), i % 25 == 0)
        start_time = time.time()
    logger.stop_timing()
    logger.synchronize_between_processes()

    # TODO: I don't think this is appropriate when use_buffers=True and using EMA (not SWA)
    # if model_ema:
    #     with torch.no_grad():
    #         copy_bn_buffers(model, model_ema.module)

    model.eval()


def evaluate(
    model: nn.Module,
    criterion: _Loss,
    data_loader: DataLoader,
    epoch: int,
    logger: MultiLogger,
    preprocess: Callable = lambda x: x,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
):
    """Evaluate the model for one validation epoch.

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
    training_state = model.training
    model.eval()
    pbar = TQDM(data_loader, desc="Evaluation", total=len(data_loader), ncols=TERMINAL_WIDTH, leave=False)
    logger.update(epoch=epoch, type="eval")
    logger.start_timing()

    num_processed_samples = 0
    start_time = time.time()
    for i, (batch, target) in enumerate(pbar):
        with torch.inference_mode():
            batch, target = batch.to(device, non_blocking=True), target.to(device, non_blocking=True)
            with autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
                output = model(preprocess(batch))
                loss = criterion(output, target)
            logger.consume(index=i, batch=batch, target=target, prediction=output, loss=loss, optimizer=None, start_time=start_time)
        pbar.set_description_str(logger.status(), i % 25 == 0)
        num_processed_samples += len(batch)
        start_time = time.time()
    logger.stop_timing()
    logger.synchronize_between_processes()

    # gather the stats from all processes
    num_processed_samples = reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and (dataset_len := len(data_loader.dataset)) != num_processed_samples # type: ignore
        and (not is_dist_avail_and_initialized() or torch.distributed.get_rank() == 0)
    ):
        log.warning(
            f"It looks like the dataset has {dataset_len} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
            "This behavior might be improved in the future."
        ) # TODO: Fixme

    if logger.verbose:
        log.info(logger.summary_string())
    logger.figures(model)

    model.train(training_state)  # Restore model state

    global_metric = logger.canonical_scalar
    if global_metric is None:
        return float('nan')
    return float(global_metric)


def train(
    model: nn.Module,
    model_ema: EMATeacher,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: _Loss,
    optimizer: Optimizer,
    scaler: GradScaler,
    lr_scheduler: LRScheduler,
    logger: MultiLogger,
    epochs: int,
    start_epoch: int = 0,
    preprocess: Callable = lambda x: x,
    augmentation: Callable = lambda x: x,
    regularizer: Callable[[nn.Module], torch.Tensor | Literal[0]] = lambda x: 0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    output_dir: str | None = None,
    weight_store_rate: int | None = None,
    **kwargs,
):
    """Full training loop across epochs with periodic evaluation and checkpointing.

    Args:
        model: Model to train.
        model_ema: Exponential Moving Average model (``AveragedModel``) linked to ``model``.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        criterion: Loss function.
        optimizer: Optimizer instance.
        scaler: Gradient scaler.
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
    log.info("Start training")
    start_time = time.time()

    eval_model: nn.Module = getattr(model_ema, "module", model)

    if is_dist_avail_and_initialized():
        log.debug("Model DDP wrap starting...")
        if "cpu" in str(device).lower():
            device_ids = None
        else:
            device_idx = device.index if (isinstance(device, torch.device) and device.index is not None) else torch.cuda.current_device()
            device_ids = [device_idx]
        model = DDP(model, device_ids=device_ids, find_unused_parameters=True)
        log.debug("DDP wrap completed.")

    best_eval_metric = -float("inf")
    best_epoch = -1
    for epoch in range(start_epoch, epochs):
        train_one_epoch(
            model,
            model_ema,
            criterion,
            optimizer,
            scaler,
            lr_scheduler,
            train_loader,
            epoch,
            logger,
            preprocess,
            augmentation,
            regularizer,
            device=device,
            dtype=dtype,
            **kwargs,
        )
        eval_metric = evaluate(eval_model, criterion, val_loader, epoch, logger, preprocess, device=device, dtype=dtype)
        is_best_eval = (best_eval_metric := max(best_eval_metric, eval_metric)) == eval_metric
        if is_best_eval:
            best_epoch = epoch
        if output_dir is not None:
            raw_model = model.module if hasattr(model, "module") else model
            assert isinstance(raw_model, nn.Module)
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler is not None:
                checkpoint["scaler"] = scaler.state_dict()
            if weight_store_rate is not None and epoch % weight_store_rate == 0:
                save_on_master(checkpoint, os.path.join(output_dir, f"checkpoint_{epoch}.pth"))
            save_on_master(checkpoint, os.path.join(output_dir, "checkpoint_last.pth"))
            if is_best_eval:
                save_on_master(eval_model.state_dict(), os.path.join(output_dir, "best.pt"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if logger.verbose:
        log.info(f"Total time {total_time_str} | Best model found at epoch {best_epoch + 1}")
        log.info(logger.timings(format=True, prefix="\n\t"))
    logger.finish()
