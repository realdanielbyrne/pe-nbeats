import os
from importlib.metadata import PackageNotFoundError, version

import torch
from .models import *
from .loaders import *
from .losses import *
from .blocks.blocks import *
from .constants import DEPRECATED_BLOCKS, get_deprecated_block_message
from .preflight import Recommendation, analyze, recommend

try:
    __version__ = version("pe-nbeats")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0+unknown"


def get_best_accelerator() -> str:
    """Detect the best available accelerator for PyTorch.

    Returns 'cuda' for NVIDIA GPUs, 'xpu' for Intel Arc (PyTorch 2.4+),
    'mps' for Apple Silicon, or 'cpu' as fallback.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Register XPU accelerator with PyTorch Lightning
# ---------------------------------------------------------------------------
# Lightning versions prior to ~2.7 do not include a native XPU accelerator.
# When torch.xpu is available we inject a minimal Accelerator subclass so
# that ``accelerator='xpu'`` is accepted by ``pl.Trainer``.
# ---------------------------------------------------------------------------

if hasattr(torch, "xpu") and torch.xpu.is_available():
    try:
        from typing import Any, Optional, Union

        from typing_extensions import override

        import lightning.pytorch as pl
        from lightning.fabric.utilities.types import _DEVICE
        from lightning.pytorch.accelerators import AcceleratorRegistry
        from lightning.pytorch.accelerators.accelerator import Accelerator

        class XPUAccelerator(Accelerator):  # type: ignore[misc]
            """Accelerator for Intel XPU devices."""

            @override
            def setup_device(self, device: torch.device) -> None:
                if device.type != "xpu":
                    raise RuntimeError(f"Device should be XPU, got {device} instead")
                if device.index is None:
                    device = torch.device("xpu", 0)
                torch.xpu.set_device(device)

            @override
            def setup(self, trainer: "pl.Trainer") -> None:
                pass

            @override
            def get_device_stats(self, device: _DEVICE) -> dict[str, Any]:
                if hasattr(torch.xpu, "memory_stats"):
                    return dict(torch.xpu.memory_stats(device))
                return {}

            @override
            def teardown(self) -> None:
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    torch.xpu.empty_cache()

            @staticmethod
            @override
            def parse_devices(
                devices: Union[int, str, list[int]],
            ) -> Optional[list[int]]:
                if isinstance(devices, int):
                    return list(range(devices))
                if isinstance(devices, str):
                    return [int(devices)]
                return devices

            @staticmethod
            @override
            def get_parallel_devices(devices: list[int]) -> list[torch.device]:
                return [torch.device("xpu", i) for i in devices]

            @staticmethod
            @override
            def auto_device_count() -> int:
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    return int(torch.xpu.device_count())
                return 0

            @staticmethod
            @override
            def is_available() -> bool:
                return hasattr(torch, "xpu") and torch.xpu.is_available()

            @staticmethod
            @override
            def name() -> str:
                return "xpu"

            @classmethod
            @override
            def register_accelerators(cls, accelerator_registry: Any) -> None:
                accelerator_registry.register(
                    cls.name(),
                    cls,
                    description=cls.__name__,
                )

        if "xpu" not in AcceleratorRegistry.available_accelerators():
            AcceleratorRegistry.register(
                "xpu", XPUAccelerator, description="XPUAccelerator"
            )
    except Exception:
        # If any Lightning import or registration fails, silently skip so we
        # don't break the package on Lightning version changes.
        pass


def __getattr__(name):
    if name in DEPRECATED_BLOCKS:
        raise AttributeError(get_deprecated_block_message(name))
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
