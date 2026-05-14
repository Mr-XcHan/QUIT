from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeviceSelection:
    requested: str
    selected: str
    cuda_available: bool
    gpu_index: int | None = None
    gpu_name: str = ""
    memory_free_mb: int | None = None
    memory_total_mb: int | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "selected": self.selected,
            "cuda_available": self.cuda_available,
            "gpu_index": self.gpu_index,
            "gpu_name": self.gpu_name,
            "memory_free_mb": self.memory_free_mb,
            "memory_total_mb": self.memory_total_mb,
            "reason": self.reason,
        }


def select_torch_device(requested: str = "auto") -> DeviceSelection:
    """Resolve user/model device strings before generated code calls torch.device().

    PyTorch does not accept `"auto"`, so CODE resolves it to `cuda:<best_index>`
    or `cpu` and patches generated experiment configs before execution.
    """
    requested = (requested or "auto").strip().lower()
    try:
        import torch
    except Exception as exc:
        return DeviceSelection(
            requested=requested,
            selected="cpu",
            cuda_available=False,
            reason=f"torch_unavailable: {exc}",
        )

    if requested not in {"auto", "gpu"}:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return DeviceSelection(requested=requested, selected="cpu", cuda_available=False, reason="cuda_requested_but_unavailable")
        return DeviceSelection(requested=requested, selected=requested, cuda_available=torch.cuda.is_available(), reason="explicit_device")

    if not torch.cuda.is_available():
        return DeviceSelection(requested=requested, selected="cpu", cuda_available=False, reason="cuda_unavailable")

    best_index = 0
    best_free = -1
    best_total = None
    for index in range(torch.cuda.device_count()):
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        except Exception:
            continue
        free_mb = int(free_bytes / (1024 * 1024))
        if free_mb > best_free:
            best_index = index
            best_free = free_mb
            best_total = int(total_bytes / (1024 * 1024))
    return DeviceSelection(
        requested=requested,
        selected=f"cuda:{best_index}",
        cuda_available=True,
        gpu_index=best_index,
        gpu_name=torch.cuda.get_device_name(best_index),
        memory_free_mb=best_free if best_free >= 0 else None,
        memory_total_mb=best_total,
        reason="selected_cuda_with_most_free_memory",
    )
