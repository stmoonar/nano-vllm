# SPDX-License-Identifier: Apache-2.0
"""
CUDA memory management for nano-vllm sleep mode.

This module implements sleep mode functionality using PyTorch's memory management.
The key insight is to:
1. Copy tensor data to CPU pinned memory
2. Release GPU tensor storage (not just delete reference)
3. Call gc.collect() and torch.cuda.empty_cache()

For wake_up:
1. Reallocate GPU tensors
2. Copy data back from CPU backup
"""
import gc
import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def is_pin_memory_available() -> bool:
    """Check if pinned memory is available."""
    return torch.cuda.is_available()


class SleepModeManager:
    """
    Manager for model sleep mode in nano-vllm.

    This provides efficient GPU memory release by:
    1. Backing up tensor data to CPU pinned memory
    2. Releasing GPU tensor storage
    3. Properly cleaning up CUDA memory caches

    Unlike simple model.to("cpu"), this approach truly releases GPU memory
    by releasing the underlying storage of tensors.
    """

    instance: "SleepModeManager | None" = None

    @staticmethod
    def get_instance() -> "SleepModeManager":
        if SleepModeManager.instance is None:
            SleepModeManager.instance = SleepModeManager()
        return SleepModeManager.instance

    def __init__(self):
        # Store CPU backups: {tag: {param_name: (cpu_tensor, original_shape, original_dtype, original_device)}}
        self.cpu_backups: dict[str, dict[str, tuple[torch.Tensor, torch.Size, torch.dtype, torch.device]]] = {}
        self.is_sleeping: bool = False
        self.sleep_level: int = 0

    def _backup_and_release_tensor(
        self,
        tensor: torch.Tensor,
        name: str,
        tag: str,
        level: int,
    ) -> int:
        """
        Backup tensor to CPU and release GPU memory.

        Returns:
            Number of bytes freed
        """
        if tensor.device.type != "cuda":
            return 0

        size_bytes = tensor.numel() * tensor.element_size()
        original_shape = tensor.shape
        original_dtype = tensor.dtype
        original_device = tensor.device

        if level == 1:
            # Level 1: Backup to CPU pinned memory
            cpu_backup = torch.empty(
                tensor.shape,
                dtype=tensor.dtype,
                device="cpu",
                pin_memory=is_pin_memory_available(),
            )
            cpu_backup.copy_(tensor)
            self.cpu_backups[tag][name] = (cpu_backup, original_shape, original_dtype, original_device)

        # Release GPU memory by resizing storage to 0
        # This is the key to truly releasing GPU memory
        tensor.data = torch.empty(0, dtype=tensor.dtype, device=tensor.device)

        return size_bytes

    def sleep_model(
        self,
        model: nn.Module,
        tag: str = "weights",
        level: int = 1,
    ) -> int:
        """
        Put model into sleep mode, releasing GPU memory.

        Args:
            model: The model to sleep
            tag: Tag for identifying this model's backup
            level: Sleep level
                - Level 1: Backup weights to CPU, can restore later
                - Level 2: Discard weights (must reload from disk)

        Returns:
            Number of bytes freed
        """
        if tag not in self.cpu_backups:
            self.cpu_backups[tag] = {}

        freed_bytes = 0

        # Backup and release parameters
        for name, param in model.named_parameters():
            freed_bytes += self._backup_and_release_tensor(
                param.data, f"param_{name}", tag, level
            )

        # Backup and release buffers
        for name, buffer in model.named_buffers():
            freed_bytes += self._backup_and_release_tensor(
                buffer, f"buffer_{name}", tag, level
            )

        self.is_sleeping = True
        self.sleep_level = level

        # Force garbage collection and CUDA cache clearing
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        logger.info(
            "SleepModeManager: Released %.2f GiB GPU memory (level=%d, tag=%s)",
            freed_bytes / 1024**3, level, tag,
        )

        return freed_bytes

    def wake_up_model(
        self,
        model: nn.Module,
        tag: str = "weights",
        device: str | torch.device = "cuda",
    ) -> None:
        """
        Wake up model from sleep mode, restoring GPU memory.

        Args:
            model: The model to wake up
            tag: Tag for identifying this model's backup
            device: Device to restore to
        """
        if tag not in self.cpu_backups:
            logger.warning(f"No backup found for tag '{tag}'")
            return

        backups = self.cpu_backups[tag]

        # Restore parameters
        for name, param in model.named_parameters():
            backup_name = f"param_{name}"
            if backup_name in backups:
                cpu_tensor, shape, dtype, orig_device = backups[backup_name]
                # Allocate new GPU tensor and copy data
                param.data = cpu_tensor.to(device)

        # Restore buffers
        for name, buffer in model.named_buffers():
            backup_name = f"buffer_{name}"
            if backup_name in backups:
                cpu_tensor, shape, dtype, orig_device = backups[backup_name]
                buffer.data = cpu_tensor.to(device)

        # Clear backup
        del self.cpu_backups[tag]

        if not self.cpu_backups:
            self.is_sleeping = False
            self.sleep_level = 0

        logger.info("SleepModeManager: Restored model to GPU (tag=%s)", tag)

    def sleep_tensor(
        self,
        tensor: torch.Tensor,
        tag: str,
        name: str,
        level: int = 1,
    ) -> int:
        """
        Put a single tensor into sleep mode.

        Args:
            tensor: The tensor to sleep
            tag: Tag for identifying this tensor's backup
            name: Name for the tensor
            level: Sleep level

        Returns:
            Number of bytes freed
        """
        if tag not in self.cpu_backups:
            self.cpu_backups[tag] = {}

        freed_bytes = self._backup_and_release_tensor(tensor, name, tag, level)

        gc.collect()
        torch.cuda.empty_cache()

        return freed_bytes

    def wake_up_tensor(
        self,
        tensor: torch.Tensor,
        tag: str,
        name: str,
        device: str | torch.device = "cuda",
    ) -> bool:
        """
        Wake up a single tensor from sleep mode.

        Args:
            tensor: The tensor to restore into
            tag: Tag for identifying this tensor's backup
            name: Name for the tensor
            device: Device to restore to

        Returns:
            True if tensor was restored, False if no backup found
        """
        if tag not in self.cpu_backups:
            return False

        backups = self.cpu_backups[tag]
        if name not in backups:
            return False

        cpu_tensor, shape, dtype, orig_device = backups[name]
        tensor.data = cpu_tensor.to(device)

        del backups[name]
        if not backups:
            del self.cpu_backups[tag]

        return True


# For backward compatibility
FallbackMemoryManager = SleepModeManager


# Simplified CuMemAllocator-like interface that doesn't require vLLM
class CuMemAllocator:
    """
    Simplified memory allocator for nano-vllm sleep mode.

    This provides a similar interface to vLLM's CuMemAllocator but uses
    PyTorch's memory management instead of CUDA virtual memory APIs.
    """

    instance: "CuMemAllocator | None" = None
    default_tag: str = "default"

    @staticmethod
    def get_instance() -> "CuMemAllocator":
        if CuMemAllocator.instance is None:
            CuMemAllocator.instance = CuMemAllocator()
        return CuMemAllocator.instance

    def __init__(self):
        self.sleep_manager = SleepModeManager.get_instance()
        self.tracked_models: dict[str, nn.Module] = {}
        self.tracked_tensors: dict[str, dict[str, torch.Tensor]] = {}
        self.current_tag: str = CuMemAllocator.default_tag

    def register_model(self, model: nn.Module, tag: str) -> None:
        """Register a model for sleep mode tracking."""
        self.tracked_models[tag] = model

    def register_tensor(self, tensor: torch.Tensor, tag: str, name: str) -> None:
        """Register a tensor for sleep mode tracking."""
        if tag not in self.tracked_tensors:
            self.tracked_tensors[tag] = {}
        self.tracked_tensors[tag][name] = tensor

    def sleep(self, offload_tags: tuple[str, ...] | str | None = None) -> int:
        """
        Put tracked models/tensors into sleep mode.

        Args:
            offload_tags: Tags to backup to CPU. Others will be discarded.

        Returns:
            Total bytes freed
        """
        if offload_tags is None:
            offload_tags = tuple(self.tracked_models.keys()) + tuple(self.tracked_tensors.keys())
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags,)

        total_freed = 0

        # Sleep models
        for tag, model in self.tracked_models.items():
            level = 1 if tag in offload_tags else 2
            total_freed += self.sleep_manager.sleep_model(model, tag, level)

        # Sleep individual tensors
        for tag, tensors in self.tracked_tensors.items():
            level = 1 if tag in offload_tags else 2
            for name, tensor in tensors.items():
                total_freed += self.sleep_manager.sleep_tensor(tensor, tag, name, level)

        return total_freed

    def wake_up(self, tags: list[str] | None = None) -> None:
        """
        Wake up tracked models/tensors from sleep mode.

        Args:
            tags: Tags to wake up. If None, wake up all.
        """
        # Wake up models
        for tag, model in self.tracked_models.items():
            if tags is None or tag in tags:
                self.sleep_manager.wake_up_model(model, tag)

        # Wake up individual tensors
        for tag, tensors in self.tracked_tensors.items():
            if tags is None or tag in tags:
                for name, tensor in tensors.items():
                    self.sleep_manager.wake_up_tensor(tensor, tag, name)

    def get_current_usage(self) -> int:
        """Get current GPU memory usage of tracked models/tensors."""
        total = 0
        for model in self.tracked_models.values():
            for param in model.parameters():
                if param.device.type == "cuda":
                    total += param.numel() * param.element_size()
            for buffer in model.buffers():
                if buffer.device.type == "cuda":
                    total += buffer.numel() * buffer.element_size()
        for tensors in self.tracked_tensors.values():
            for tensor in tensors.values():
                if tensor.device.type == "cuda":
                    total += tensor.numel() * tensor.element_size()
        return total


# Flag to indicate cumem is available (always True for this implementation)
cumem_available = True
