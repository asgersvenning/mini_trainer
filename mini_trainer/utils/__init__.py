import copy
import hashlib
import math
import os
import shutil
import tempfile
from collections import OrderedDict
from glob import glob
from typing import Any, Callable, Iterable, Optional

import psutil
import torch
from torch import distributed as dist
from torchvision.transforms.v2 import ToDtype


def make_convert_dtype(dtype : torch.dtype, scale : bool=True):
    """
    See `torchvision.transforms.v2.ToDtype`.
    """
    return ToDtype(dtype=dtype, scale=scale)

TERMINAL_WIDTH, _ = shutil.get_terminal_size()

def float_signif_decimal(value : float, digits : int=3):
    if value == 0 or not math.isfinite(value):
        return 0
    min_b10 = math.log10(abs(value))
    if abs(min_b10 - abs(min_b10)) < 1e-1:
        min_b10 = round(min_b10)
    else:
        min_b10 = math.floor(min_b10)
    return -min(-1, min_b10 - digits + 1)

def decimals(value : float, tol : int=6):
    fv = f'{value}'
    if "." not in fv:
        return 0
    dec = fv.index(".") + 1
    for d, i in enumerate(range(dec, len(fv))):
        trunc_value = float(fv[:i])
        if abs(value - trunc_value) < 10**(-(i+tol)):
            return d
    return d + 1

def memory_proportion(
        shape : tuple[int, ...], 
        device : torch.types.Device, 
        dtype : torch.dtype
    ):
    numel = math.prod(shape)
    # bytes per element
    bpe = torch.empty(0, dtype=dtype).element_size()
    required = numel * bpe

    if isinstance(device, str) and "disk" in device.lower().strip():
        free = shutil.disk_usage(tempfile.gettempdir()).free
    else:
        dev = torch.device(device)
        if dev.type == 'cuda':
            free, _ = torch.cuda.mem_get_info(dev)
        else:
            free = psutil.virtual_memory().available

    return required / free

def increment_name_dir(name : str, dir : Optional[str]=None, max_iter : int=1000):
    if name is None:
        raise ValueError('A model name must be specified.')
    if not isinstance(name, str):
        raise TypeError(f'Invalid type `{type(name)}` used for the model name. Only `str` is accepted.')
    if len(name) == 0:
        raise ValueError('Invalid zero-length model specified.')
    def _name(i : int):
        if i < 0:
            raise RuntimeError(f'Invalid name iteration {i} specified.')
        if i == 0:
            return name
        return f'{name}_{i}'

    fs = set([os.path.splitext(os.path.basename(f))[0] for f in glob(name + "*", root_dir=dir)])
    for i in range(max_iter+1):
        if (this := _name(i)) not in fs:
            return this
    
    raise RuntimeError(f'Unable to create a new model name from {name} in {dir}, the maximum number of model iterations with the same base name {max_iter} has been reached. OBS: The name check is file-extension agnostic!')

def recursive_dfs_attr(obj: Any, attr: str,
                       predicate: Callable[[Any], bool] = lambda x: True) -> Any:
    seen = set()
    stack = [obj]
    while stack:
        current = stack.pop()
        obj_id = id(current)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        if hasattr(current, attr):
            value = getattr(current, attr)
            if predicate(value):
                return value
        if isinstance(current, Iterable) and not isinstance(current, (str, bytes, bytearray)):
            try:
                stack.extend(current)
            except TypeError:
                pass  # non-iterable despite isinstance claiming so (rare, but safe)
    raise StopIteration(f"No attribute '{attr}' found passing predicate.")

def cosine_schedule_with_warmup(total : int, warmup : int, start : float, end : float):
    def _shape_fn(step : int):
        if warmup > 0 and step < warmup:
            # linear warm-up from start_factor -> 1.0
            return start + (1.0 - start) * step / warmup
        # cosine decay from 1.0 -> min_factor
        progress = (step - warmup) / max(1, total - warmup)
        return end + 0.5 * (1.0 - end) * (1.0 + math.cos(math.pi * progress))
    return _shape_fn


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)

def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t

def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights. Original implementation taken from:
    https://github.com/pytorch/fairseq/blob/a48f235636557b8d3bc4922a6fa90f3a0fa57955/scripts/average_checkpoints.py#L16

    Args:
      inputs (list[str]): An iterable of string paths of checkpoints to load from.
    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)
    for fpath in inputs:
        with open(fpath, "rb") as f:
            state = torch.load(
                f, map_location=(lambda s, _: torch.serialization.default_restore_location(s, "cpu")), weights_only=True
            )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state
        model_params = state["model"]
        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                f"For checkpoint {f}, expected list of params: {params_keys}, but found: {model_params_keys}"
            )
        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p
    averaged_params = OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    new_state["model"] = averaged_params
    return new_state


def store_model_weights(model, checkpoint_path, checkpoint_key="model", strict=True):
    """
    This method can be used to prepare weights files for new models. It receives as
    input a model architecture and a checkpoint from the training script and produces
    a file with the weights ready for release.

    Examples:
        from torchvision import models as M

        # Classification
        model = M.mobilenet_v3_large(weights=None)
        print(store_model_weights(model, './class.pth'))

        # Quantized Classification
        model = M.quantization.mobilenet_v3_large(weights=None, quantize=False)
        model.fuse_model(is_qat=True)
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
        _ = torch.ao.quantization.prepare_qat(model, inplace=True)
        print(store_model_weights(model, './qat.pth'))

        # Object Detection
        model = M.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)
        print(store_model_weights(model, './obj.pth'))

        # Segmentation
        model = M.segmentation.deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None, aux_loss=True)
        print(store_model_weights(model, './segm.pth', strict=False))

    Args:
        model (pytorch.nn.Module): The model on which the weights will be loaded for validation purposes.
        checkpoint_path (str): The path of the checkpoint we will load.
        checkpoint_key (str, optional): The key of the checkpoint where the model weights are stored.
            Default: "model".
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

    Returns:
        output_path (str): The location where the weights are saved.
    """
    # Store the new model next to the checkpoint_path
    checkpoint_path = os.path.abspath(checkpoint_path)
    output_dir = os.path.dirname(checkpoint_path)

    # Deep copy to avoid side effects on the model object.
    model = copy.deepcopy(model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Load the weights to the model to validate that everything works
    # and remove unnecessary weights (such as auxiliaries, etc.)
    if checkpoint_key == "model_ema":
        del checkpoint[checkpoint_key]["n_averaged"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint[checkpoint_key], "module.")
    model.load_state_dict(checkpoint[checkpoint_key], strict=strict)

    tmp_path = os.path.join(output_dir, str(model.__hash__()))
    torch.save(model.state_dict(), tmp_path)

    sha256_hash = hashlib.sha256()
    with open(tmp_path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
        hh = sha256_hash.hexdigest()

    output_path = os.path.join(output_dir, "weights-" + str(hh[:8]) + ".pth")
    os.replace(tmp_path, output_path)

    return output_path

def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[list[type]] = None,
    custom_keys_weight_decay: Optional[list[tuple[str, float]]] = None,
):
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
    }
    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            for key in custom_keys:
                target_name = f"{prefix}.{name}" if prefix != "" and "." in key else name
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append({"params": params[key], "weight_decay": params_weight_decay[key]})
    return param_groups