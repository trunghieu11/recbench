"""Memory planning for dense item x item models (EASE, Turbo-CF). No torch import at module level."""

from __future__ import annotations

from typing import Any


def dense_item_cap(configured: int, device: Any = None, n_matrices: int = 3, bytes_per_value: int = 4, share: float = 0.8) -> int:
    """How many items a dense item x item float32 model can keep.

    On a CPU this is the configured cap. On a GPU it is lowered to what `n_matrices` such matrices fit in
    `share` of the free memory: n = sqrt(share * free / (n_matrices * 4 bytes)). On a 48 GB card with
    3 matrices, that is about 55,000 items.
    """
    if getattr(device, "type", str(device)) != "cuda":
        return configured
    import torch

    free, _ = torch.cuda.mem_get_info(device)
    return max(1000, min(configured, int((share * free / (n_matrices * bytes_per_value)) ** 0.5)))
