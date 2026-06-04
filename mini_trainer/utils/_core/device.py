import torch


def detect_available_devices() -> list[str]:
    """Detect all available hardware device types on the current system."""
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def select_best_device(devices: list[str] | None = None) -> str:
    """Select the best available device type from a list of devices or the system."""
    if devices is None:
        devices = detect_available_devices()
    for dev in ["cuda", "mps", "cpu"]:
        if dev in devices:
            return dev
    return "cpu"


def validate_device(device: str | torch.device) -> None:
    """Validate that the requested device is available on the system."""
    device_obj = torch.device(device) if isinstance(device, str) else device
    if device_obj.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is not available on this system, but device '{device}' was requested.")
        device_index = device_obj.index if device_obj.index is not None else 0
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested CUDA device '{device}' is not available. System has only {torch.cuda.device_count()} CUDA device(s)."
            )
    elif device_obj.type == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError(f"MPS is not available on this system, but device '{device}' was requested.")


def resolve_device(device: str | torch.device) -> torch.device:
    """Resolve an ambiguous device string or object and validate its availability."""
    device_obj = torch.device(device) if isinstance(device, str) else device
    if device_obj.type == "cuda" and device_obj.index is None:
        if torch.cuda.is_available():
            device_obj = torch.device(f"cuda:{torch.cuda.current_device()}")
    validate_device(device_obj)
    return device_obj


def setup_device(device: str | torch.device) -> torch.device:
    """Resolve, validate, and configure precision settings for the target device."""
    device_obj = resolve_device(device)
    if device_obj.type == "cuda":
        torch.set_float32_matmul_precision("high")
    return device_obj
