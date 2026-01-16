import datetime
import os
import time
from statistics import mean
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
from mini_trainer.builders import EMATeacher
from mini_trainer.utils import (TERMINAL_WIDTH, is_dist_avail_and_initialized,
                                reduce_across_processes, save_on_master)
from mini_trainer.utils.loader import AdaptiveDataLoader
from mini_trainer.utils.logging import MultiLogger


@torch.no_grad()
def EL2N(logits : torch.Tensor, labels : torch.Tensor):
    out = []
    for lo, la in zip(logits, labels):
        lo = lo.detach().clone().softmax(0)
        lo[la] -= 1
        out.append((lo ** 2).sum().item())
    return out

SQRT2 = 2**(1/2)

@torch.no_grad()
def CE_2_EL2N(loss : list[torch.Tensor]):
    return [mean(vs) for vs in zip(*[(SQRT2 * (1 - (-term.detach()/2).exp())).tolist() for term in loss])]

def train_one_epoch(
        model : nn.Module, 
        model_ema : EMATeacher,
        criterion : _Loss, 
        optimizer : Optimizer, 
        scaler : GradScaler,
        lr_scheduler : LRScheduler,
        data_loader : AdaptiveDataLoader, 
        epoch : int, 
        logger : MultiLogger,
        preprocess : Callable=lambda x : x,
        augmentation : Callable=lambda x : x,
        regularizer : Callable[[nn.Module], torch.Tensor]=lambda _: 0.,
        clip_grad_norm : float | None=5,
        device : torch.types.Device=torch.device("cpu"),
        dtype : torch.dtype=torch.float32,
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
    model.train()
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
            raise RuntimeError(f'Incorrect {batch.shape=}, expected 4 dimensions, not {len(batch.shape)}.')
        batch, target = batch.to(device), target.to(device)
        with autocast(device_type=device.type, dtype=dtype):
            logits = model(preprocess(augmentation(batch)))
            loss : list[torch.Tensor] | torch.Tensor = criterion(logits, target)
            # If EMA is disabled ``distill_loss`` is ``0.0``
            distill_loss = model_ema.teach(
                step=step,
                input=preprocess(batch),
                student=logits
            )
            reg = regularizer(model)
        if isinstance(loss, torch.Tensor):
            loss = [loss]
        prior = CE_2_EL2N(loss)
        data_loader.update(prior)
        loss = [l.mean() for l in loss]
        optimizer.zero_grad()
        if not all([torch.isfinite(term).all() for term in loss]) or not torch.isfinite(torch.as_tensor(distill_loss)):
            nan_errs += 1
            if nan_errs < 5:
                continue
            else:
                raise RuntimeError('Interrupted training due to persistent nan\'s detected in the loss.')
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
        _any_opt_stepped = (_scale <= scaler.get_scale())
        if _any_opt_stepped:
            model_ema.update_parameters(step, model)
            lr_scheduler.step()
        
        logger.consume(
            index=i,
            batch=batch, 
            target=target, 
            prediction=logits, 
            loss=loss, 
            optimizer=optimizer, 
            start_time=start_time,
            distillation_loss=distill_loss if isinstance(distill_loss, float) else float(distill_loss.detach().item()),
            regularization=reg if isinstance(reg, float) else float(reg.detach().item())
        )
        pbar.set_description_str(logger.status(), i % 25 == 0)
        start_time = time.time()
    logger.stop_timing()
    # TODO: I don't think this is appropriate when use_buffers=True and using EMA (not SWA)
    # if model_ema:
    #     with torch.no_grad():
    #         copy_bn_buffers(model, model_ema.module)

    model.eval()


def evaluate(
        model : nn.Module, 
        criterion : _Loss, 
        data_loader : AdaptiveDataLoader, 
        epoch : int,
        logger : MultiLogger,
        preprocess : Callable=lambda x : x,
        device : torch.types.Device=torch.device("cpu"),
        dtype : torch.dtype=torch.float32
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
            with autocast(device_type=device.type, dtype=dtype):
                output = model(preprocess(batch))
                loss = criterion(output, target)
            if isinstance(loss, torch.Tensor):
                loss = [loss]
            data_loader.update([1.0 for _ in range(len(batch))])
            loss = [l.mean() for l in loss]
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
    logger.stop_timing()
    
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
        print(logger.summary_string())
    logger.figures(model)
    
    model.train(training_state) # Restore model state
    
    return float(logger.canonical_scalar)


def train(
        model : nn.Module, 
        model_ema : EMATeacher,
        train_loader : DataLoader, 
        val_loader : DataLoader,
        criterion : _Loss, 
        optimizer : Optimizer, 
        scaler : GradScaler,
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
    if logger.verbose:
        print("Start training")
    start_time = time.time()

    eval_model : nn.Module = getattr(model_ema, "module", model)

    best_eval_metric = -float("inf")
    best_epoch = -1
    for epoch in range(start_epoch, epochs):
        train_one_epoch(
            model, model_ema, criterion, optimizer, scaler, lr_scheduler, 
            train_loader, epoch, logger, preprocess, augmentation, regularizer, 
            device=device, dtype=dtype, **kwargs
        )
        eval_metric = evaluate(eval_model, criterion, val_loader, epoch, logger, preprocess, device=device, dtype=dtype)
        is_best_eval = (best_eval_metric := max(best_eval_metric, eval_metric)) == eval_metric
        if is_best_eval:
            best_epoch = epoch
        if output_dir is not None:
            checkpoint = {
                "model": model.state_dict(),
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
        print(f"Total time {total_time_str} | Best model found at epoch {best_epoch + 1}")
        print(logger.timings(format=True, prefix="\n\t"))
    logger.finish()
