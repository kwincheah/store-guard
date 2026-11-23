"""CPU/GPU device resolution helpers for ONNX Runtime and PyTorch/Ultralytics."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


def normalize_device_request(device: str = "auto") -> str:
    req = str(device or "auto").strip().lower() or "auto"
    if req in ("auto", "cpu", "cuda", "mps"):
        return "cuda:0" if req == "cuda" else req
    if req.startswith("cuda:"):
        return req
    return "auto"


def get_device_capabilities(requested_device: str = "auto") -> Dict[str, Any]:
    """Inspect ONNX/Torch hardware capabilities and describe active device resolution."""
    req = normalize_device_request(requested_device)
    try:
        import onnxruntime as ort
        available_providers = list(ort.get_available_providers())
    except Exception:
        available_providers = []

    has_ort_cuda = "CUDAExecutionProvider" in available_providers
    try:
        import torch
        has_torch_cuda = bool(torch.cuda.is_available())
    except Exception:
        has_torch_cuda = False

    _, using_gpu, resolved_label = resolve_onnx_providers(req)
    cuda_available = bool(has_ort_cuda or has_torch_cuda)
    fallback_note = ""
    if req.startswith("cuda") and not using_gpu:
        fallback_note = "CUDA GPU provider unavailable (install onnxruntime-gpu + NVIDIA CUDA driver); running on CPU."

    return {
        "requested_device": req,
        "resolved_device": resolved_label,
        "using_gpu": using_gpu,
        "cuda_available": cuda_available,
        "available_providers": available_providers,
        "fallback_note": fallback_note,
    }


def resolve_onnx_providers(device: str = "auto") -> Tuple[List[Any], bool, str]:
    """Resolve 'auto'/'cpu'/'cuda'/'cuda:N' into (providers, using_gpu, resolved_label)."""
    req = str(device or "auto").strip().lower() or "auto"
    try:
        import onnxruntime as ort
        available = list(ort.get_available_providers())
    except Exception:
        available = []

    has_cuda = "CUDAExecutionProvider" in available
    if req == "cpu":
        return ["CPUExecutionProvider"], False, "cpu"

    if (req == "auto" or req.startswith("cuda")) and has_cuda:
        dev_id = 0
        if ":" in req:
            try:
                dev_id = max(0, int(req.split(":", 1)[1]))
            except (TypeError, ValueError):
                dev_id = 0
        return [("CUDAExecutionProvider", {"device_id": dev_id}), "CPUExecutionProvider"], True, f"cuda:{dev_id}"

    return ["CPUExecutionProvider"], False, "cpu"


def resolve_torch_device(device: str = "auto") -> str:
    """Resolve 'auto'/'cpu'/'cuda'/'cuda:N'/'mps' to an available PyTorch device string."""
    req = str(device or "auto").strip().lower() or "auto"
    try:
        import torch
        has_cuda = bool(torch.cuda.is_available())
        has_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        has_cuda = False
        has_mps = False

    if req == "auto":
        if has_cuda:
            return "cuda:0"
        if has_mps:
            return "mps"
        return "cpu"
    if req.startswith("cuda"):
        return req if has_cuda else "cpu"
    if req == "mps":
        return "mps" if has_mps else "cpu"
    return "cpu"
