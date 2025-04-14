# Courtesy of: https://github.com/pytorch/vision/blob/main/references/classification/utils.py

import copy
import datetime
import errno
import glob
import hashlib
import json
import os
import random
import time
from collections import OrderedDict, defaultdict, deque
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import ConvertImageDtype
from torchvision.transforms.functional import resize

convert2fp16 = ConvertImageDtype(torch.float16)
convert2bf16 = ConvertImageDtype(torch.bfloat16)
convert2fp32 = ConvertImageDtype(torch.float32)
convert2uint8 = ConvertImageDtype(torch.uint8)

def parse_class_index(path : Optional[str]=None, dir : Optional[str]=None):
    """
    Accepts a path to 
    """
    if not os.path.exists(path):
        if dir is None or not os.path.isdir(dir):
            raise TypeError(f'If `path` is not the path to a valid file, `dir` must be a valid directory, not \'{dir}\'.')
        cls2idx = {cls : i for i, cls in enumerate(sorted([f for f in map(os.path.basename, os.listdir(dir)) if os.path.isdir(os.path.join(dir, f))]))}
        if path is not None:
            with open(path, "w") as f:
                json.dump(cls2idx, f)
    else:
        with open(path, "rb") as f:
            cls2idx = json.load(f)
    cls = list(cls2idx.keys())
    idx2cls = {v : k for k, v in cls2idx.items()}
    ncls = len(idx2cls)
    return {"num_classes" : ncls}, {"classes" : cls, "class2idx" : cls2idx, "idx2class" : idx2cls}

def write_metadata(directory : str, classes : List[str], dst : str, train_proportion : float=0.9):
    data = {
        "path" : [],
        "class" : [],
        "split" : []
    }
    for cls in classes:
        this_dir = os.path.join(directory, cls)
        for file in map(os.path.basename, os.listdir(this_dir)):
            data["path"].append(os.path.join(this_dir, file))
            data["class"].append(cls)
            data["split"].append("train" if random.random() < train_proportion else "validation")
    with open(dst, "w") as f:
        json.dump(data, f)

def is_image(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            header = f.read(16)  # read enough bytes for JPEG and PNG signatures
    except Exception:
        return False

    # JPEG files start with: 0xFF, 0xD8
    if header.startswith(b'\xff\xd8'):
        return True

    # PNG files start with: 0x89, 'PNG', CR, LF, 0x1A, LF
    if header.startswith(b'\x89PNG\r\n\x1a\n'):
        return True

    return False

def get_image_data(path : str):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Meta data file ({path}) for training split not found. Please provide a JSON with the following keys: "path", "class", "split".')
        with open(path, "rb") as f:
            _image_data = {k : np.array(v) for k, v in json.load(f).items()}
        image_data = {k : v[np.array([is_image(f) for f in _image_data["path"]])] for k, v in _image_data.items()}
        train_image_data = {k : v[image_data["split"] == np.array("train")] for k, v in image_data.items()}
        val_image_data = {k : v[image_data["split"] == np.array("validation")] for k, v in image_data.items()}
        return train_image_data, val_image_data

def find_images(root : str):
    paths = glob.glob(os.path.join(root, "**"), recursive=True)
    return list(filter(is_image, paths))

class LazyDataset(Dataset):
    def __init__(self, func : Callable[[str], torch.Tensor], items : list[str]):
        self.func = func
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.func(self.items[i])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

class ImageLoader:
    def __init__(self, preprocessor, dtype, device):
        self.dtype, self.device = dtype, device
        self.preprocessor = preprocessor
        match self.dtype:
            case torch.float16:
                self.converter = convert2fp16
            case torch.float32:
                self.converter = convert2fp32
            case torch.bfloat16:
                self.converter = convert2bf16
            case _:
                raise ValueError("Only fp16 supported for now.")
        size = self.preprocessor.resize_size if hasattr(self.preprocessor, "resize_size") else 256
        self.shape = size if not isinstance(size, int) and len(size) == 2 else (size, size)
    
    def __call__(self, x : Union[str, Iterable]):
        if isinstance(x, str):
            proc_img : torch.Tensor = self.preprocessor(resize(self.converter(decode_image(x, ImageReadMode.RGB)), self.shape)).to(self.device)
            return proc_img
        return LazyDataset(self, x)
    
class ImageClassLoader:
    def __init__(self, class_decoder, preprocessor, dtype, device):
        self.dtype, self.device = dtype, device
        self.preprocessor = preprocessor
        self.class_decoder = class_decoder
        match self.dtype:
            case torch.float16:
                self.converter = convert2fp16
            case torch.float32:
                self.converter = convert2fp32
            case torch.bfloat16:
                self.converter = convert2bf16
            case _:
                raise ValueError("Only fp16 supported for now.")
        size = self.preprocessor.resize_size if hasattr(self.preprocessor, "resize_size") else 256
        self.shape = size if not isinstance(size, int) and len(size) == 2 else (size, size)
    
    def __call__(self, x : Union[str, Iterable]):
        if isinstance(x, str):
            proc_img : torch.Tensor = self.preprocessor(resize(self.converter(decode_image(x, ImageReadMode.RGB)), self.shape)).to(self.device)
            cls = self.class_decoder(x)
            return proc_img, cls
        return LazyDataset(self, x)

def confusion_matrix(results : Dict[str, List[str]], i2c : Dict[int, str], keys : Tuple[str, str]=("pred", "gt")):
    # Build confusion matrix and compute accuracies
    classes = [i2c[i] for i in sorted(list(i2c))]

    # Initialize confusion matrix and counters
    conf_mat = {gt: {pred: 0 for pred in classes} for gt in classes}
    total_correct = 0
    total_samples = 0
    per_class_total = {cls: 0 for cls in classes}
    per_class_correct = {cls: 0 for cls in classes}

    # Populate confusion matrix and count correct predictions
    for p, g in zip(results[keys[0]], results[keys[1]]):
        conf_mat[g][p] += 1
        total_samples += 1
        per_class_total[g] += 1
        if g.lower().strip() == p.lower().strip():
            total_correct += 1
            per_class_correct[g] += 1

    # Print the confusion matrix (numbers only, aligned)
    max_cf_n = max(val for d in conf_mat.values() for val in d.values())
    width = len(str(max_cf_n))
    for gt in classes:
        row_str = "|".join(
            "{:>{width}d}".format(conf_mat[gt][pred], width=width)
            if conf_mat[gt][pred] != 0
            else " " * width
            for pred in classes
        )
        print(row_str)

    # Compute and print per-class accuracies
    print("\nPer-class Accuracies:")
    macro_acc = 0.0
    num_class_with_any = 0
    for cls in classes:
        if per_class_total[cls] > 0:
            acc = per_class_correct[cls] / per_class_total[cls]
            num_class_with_any += 1
        else:
            acc = 0.0
        macro_acc += acc
        print(f"{cls:_<{max(map(len, classes))}}{acc:_>9.1%} ({per_class_correct[cls]}/{per_class_total[cls]})")
    macro_acc /= num_class_with_any or 1

    # Micro accuracy: overall correct predictions / total predictions
    micro_acc = total_correct / total_samples if total_samples > 0 else 0.0

    print(f"\nMicro Accuracy: {micro_acc:.2%} ({total_correct}/{total_samples})")
    print(f"Macro Accuracy: {macro_acc:.2%}")

class _ResultsCollector:
    def collect(
            self, 
            paths : List[str],
            prediction : Any,
            *args, 
            **kwargs
        ):
        raise NotImplementedError('Result collectors must have a `collect` class method.')
    
    def evaluate(self):
        raise NotImplementedError('Result collector must have a `evaluate` class method.')
    
    @property
    def data(self):
        raise NotImplementedError('Result collector must have `data` class propery suitable for JSON serialization.')

class BaseResultCollector(_ResultsCollector):
    def __init__(self, idx2class : Dict[int, str], training_format : bool=False, *args, **kwargs):
        self.paths = []
        self.preds = []
        self.confs = []
        self.idx2class = idx2class
        self.training_format = training_format
    
    def collect(
            self,
            paths,
            predictions
        ):
        self.paths.extend(paths)
        self.preds.extend([self.idx2class[idx] for idx in predictions.argmax(1).tolist()])
        self.confs.extend(predictions.softmax(1).max(0).values.tolist())

    def evaluate(self):
        if self.training_format:
            data = self.data
            data["gt"] = [f.split(os.sep)[-2] for f in data["path"]]

            confusion_matrix(data, self.idx2class)
    
    @property
    def data(self):
        return {
            "path" : self.paths,
            "pred" : self.preds,
            "conf" : self.confs
        }

class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = reduce_across_processes([self.count, self.total])
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )

def debug_augmentation(
        augmentation,
        dataset,
        strict : bool=True
    ):
    try:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        example_image = dataset[random.choice(range(len(dataset)))][0].clone().float().cpu()
        
        axs[0].imshow(example_image.permute(1,2,0))
        axs[1].imshow(augmentation(example_image).permute(1,2,0))

        plt.savefig("example_augmentation.png")
        plt.close()
    except Exception as e:
        e_msg = (
            "Error while attempting to create debug augmentation image."
            "Perhaps the supplied dataloader doesn't return items (image, label) in the expected format."
        )
        e.add_note(e_msg)
        if strict:
            raise e
        print(e_msg)
        return False
    return True

class MetricLogger:
    def __init__(self, delimiter="\t", printer=print):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.printer = printer

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    self.printer(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    self.printer(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


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
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights. Original implementation taken from:
    https://github.com/pytorch/fairseq/blob/a48f235636557b8d3bc4922a6fa90f3a0fa57955/scripts/average_checkpoints.py#L16

    Args:
      inputs (List[str]): An iterable of string paths of checkpoints to load from.
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


def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t


def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[type]] = None,
    custom_keys_weight_decay: Optional[List[Tuple[str, float]]] = None,
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