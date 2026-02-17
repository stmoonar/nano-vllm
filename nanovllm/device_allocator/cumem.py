# SPDX-License-Identifier: Apache-2.0
"""
cumem-based pytorch pluggable allocator to implement sleep mode for nano-vllm.

This module provides a simplified but functional implementation of vLLM's sleep mode.
It uses CUDA virtual memory APIs (via vLLM's cumem_allocator C++ extension if available)
or falls back to a pure PyTorch implementation.

Sleep mode allows temporarily freeing GPU memory while keeping the model state,
useful for scenarios like:
- Running multiple models on the same GPU
- RLHF workflows where reward models are loaded/unloaded
- Memory-constrained inference scenarios
"""
import dataclasses
import gc
import logging
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch

logger = logging.getLogger(__name__)


# Try to import vLLM's cumem allocator (if available)
cumem_available = False
try:
    from vllm.cumem_allocator import (
        init_module,
        python_create_and_map,
        python_unmap_and_release,
    )
    from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
    from vllm.utils.system_utils import find_loaded_library

    lib_name = find_loaded_library("cumem_allocator")
    libcudart = CudaRTLibrary()
    cumem_available = True
    logger.info("Using vLLM's cumem allocator for sleep mode")
except (ModuleNotFoundError, ImportError):
    # vLLM cumem not available, will use fallback
    init_module = None
    python_create_and_map = None
    python_unmap_and_release = None
    CudaRTLibrary = None
    lib_name = None
    libcudart = None
    logger.info("vLLM cumem not available, using fallback implementation")


def is_pin_memory_available() -> bool:
    """Check if pinned memory is available."""
    return torch.cuda.is_available()


# py_device, py_alignedSize, py_d_mem, py_p_memHandle
HandleType = tuple[int, int, int, int]


@dataclasses.dataclass
class AllocationData:
    """Data structure to track memory allocations."""
    handle: HandleType | None
    tag: str
    cpu_backup_tensor: torch.Tensor | None = None
    # For fallback mode: track original tensor info
    original_data_ptr: int | None = None
    size_bytes: int = 0


def create_and_map(allocation_handle: HandleType) -> None:
    """Create and map GPU memory."""
    if cumem_available:
        python_create_and_map(*allocation_handle)


def unmap_and_release(allocation_handle: HandleType) -> None:
    """Unmap and release GPU memory."""
    if cumem_available:
        python_unmap_and_release(*allocation_handle)


def get_pluggable_allocator(
    python_malloc_fn: Callable[[HandleType], None],
    python_free_func: Callable[[int], HandleType]
) -> "torch.cuda.memory.CUDAPluggableAllocator":
    """Get a pluggable allocator with custom callbacks."""
    init_module(python_malloc_fn, python_free_func)
    new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
        lib_name, "my_malloc", "my_free"
    )
    return new_alloc


@contextmanager
def use_memory_pool_with_allocator(
    python_malloc_fn: Callable[[HandleType], None],
    python_free_func: Callable[[int], HandleType]
):
    """Context manager to use a memory pool with custom allocator."""
    new_alloc = get_pluggable_allocator(python_malloc_fn, python_free_func)
    mem_pool = torch.cuda.memory.MemPool(new_alloc._allocator)
    with torch.cuda.memory.use_mem_pool(mem_pool):
        yield mem_pool, new_alloc


class CuMemAllocator:
    """
    A singleton class that manages a memory pool for CUDA tensors.
    The memory in this pool can be offloaded or discarded when the
    allocator sleeps.

    Inside the `use_memory_pool(tag)` context, all tensors created will
    be allocated in the memory pool, and has the same tag as the
    tag passed to the context.

    When we call `sleep`, all tensors with the specified tag will be
    offloaded to CPU memory, and the rest of the tensors will be discarded.
    When we call `wake_up`, all tensors that are previously offloaded
    will be loaded back to GPU memory, and the rest of the tensors will
    have empty memory.

    This implementation supports two modes:
    1. cumem mode: Uses vLLM's C++ cumem allocator (preferred)
    2. fallback mode: Pure PyTorch implementation for compatibility
    """

    instance: "CuMemAllocator | None" = None
    default_tag: str = "default"

    @staticmethod
    def get_instance() -> "CuMemAllocator":
        """
        CuMemAllocator is a singleton class.
        We cannot call the constructor directly.
        Call this method to get the instance.
        """
        if CuMemAllocator.instance is None:
            CuMemAllocator.instance = CuMemAllocator()
        return CuMemAllocator.instance

    def __init__(self):
        self.pointer_to_data: dict[int, AllocationData] = {}
        self.current_tag: str = CuMemAllocator.default_tag
        self.allocator_and_pools: dict[str, Any] = {}
        self.use_cumem = cumem_available

        # For fallback mode: track tensors directly
        self.fallback_tensors: dict[str, dict[int, torch.Tensor]] = {}
        self.is_sleeping = False

        # Creating strong references to the two callbacks here to prevent
        # these ephemeral bound-method objects being garbage collected.
        if self.use_cumem:
            self.python_malloc_callback = self._python_malloc_callback
            self.python_free_callback = self._python_free_callback

    def _python_malloc_callback(self, allocation_handle: HandleType) -> None:
        """
        Internal method to store the allocation data
        when memory is allocated in the memory pool.
        """
        py_d_mem = allocation_handle[2]
        self.pointer_to_data[py_d_mem] = AllocationData(
            allocation_handle, self.current_tag
        )
        logger.debug(
            "Allocated %s bytes for %s with address %s from cumem allocator",
            allocation_handle[1],
            self.current_tag,
            py_d_mem,
        )

    def _python_free_callback(self, ptr: int) -> HandleType:
        """
        Internal method to look up the allocation data
        when memory is freed in the memory pool.
        """
        data = self.pointer_to_data.pop(ptr)
        if data.cpu_backup_tensor is not None:
            data.cpu_backup_tensor = None
        logger.debug(
            "Freed %s bytes for %s with address %s from cumem allocator",
            data.handle[1] if data.handle else 0,
            data.tag,
            ptr,
        )
        return data.handle

    def register_tensor(self, tensor: torch.Tensor, tag: str | None = None) -> None:
        """
        Register a tensor for tracking (fallback mode).
        This allows tracking tensors that weren't created within use_memory_pool.
        """
        if tag is None:
            tag = self.current_tag

        if tag not in self.fallback_tensors:
            self.fallback_tensors[tag] = {}

        ptr = tensor.data_ptr()
        self.fallback_tensors[tag][ptr] = tensor

        # Also track in pointer_to_data for consistency
        self.pointer_to_data[ptr] = AllocationData(
            handle=None,
            tag=tag,
            original_data_ptr=ptr,
            size_bytes=tensor.numel() * tensor.element_size(),
        )

    def sleep(self, offload_tags: tuple[str, ...] | str | None = None) -> None:
        """
        Put the allocator in sleep mode.
        All data in the memory allocation with the specified tag will be
        offloaded to CPU memory, and others will be discarded.

        :param offload_tags: The tags of the memory allocation that will be
            offloaded. The rest of the memory allocation will be discarded.
        """
        if offload_tags is None:
            offload_tags = (CuMemAllocator.default_tag,)
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags,)

        assert isinstance(offload_tags, tuple)

        total_bytes = 0
        backup_bytes = 0

        if self.use_cumem:
            # Use cumem-based sleep
            for ptr, data in self.pointer_to_data.items():
                handle = data.handle
                if handle is None:
                    continue
                total_bytes += handle[1]
                if data.tag in offload_tags:
                    backup_bytes += handle[1]
                    size_in_bytes = handle[1]
                    cpu_backup_tensor = torch.empty(
                        size_in_bytes,
                        dtype=torch.uint8,
                        device="cpu",
                        pin_memory=is_pin_memory_available(),
                    )
                    cpu_ptr = cpu_backup_tensor.data_ptr()
                    libcudart.cudaMemcpy(cpu_ptr, ptr, size_in_bytes)
                    data.cpu_backup_tensor = cpu_backup_tensor
                unmap_and_release(handle)
        else:
            # Fallback mode: offload tensors to CPU
            for tag, tensors in self.fallback_tensors.items():
                for ptr, tensor in tensors.items():
                    size_bytes = tensor.numel() * tensor.element_size()
                    total_bytes += size_bytes
                    data = self.pointer_to_data.get(ptr)
                    if data is None:
                        continue
                    if tag in offload_tags:
                        backup_bytes += size_bytes
                        # Create CPU backup
                        cpu_backup = tensor.cpu().clone()
                        if is_pin_memory_available():
                            cpu_backup = cpu_backup.pin_memory()
                        data.cpu_backup_tensor = cpu_backup
                    # Clear GPU tensor data (but keep the tensor shell)
                    tensor.data = torch.empty(0, device=tensor.device, dtype=tensor.dtype)

        self.is_sleeping = True

        logger.info(
            "CuMemAllocator: sleep freed %.2f GiB memory in total, of which "
            "%.2f GiB is backed up in CPU and the rest %.2f GiB is discarded "
            "directly.",
            total_bytes / 1024**3,
            backup_bytes / 1024**3,
            (total_bytes - backup_bytes) / 1024**3,
        )

        gc.collect()
        torch.cuda.empty_cache()

    def wake_up(self, tags: list[str] | None = None) -> None:
        """
        Wake up the allocator from sleep mode.
        All data that is previously offloaded will be loaded back to GPU
        memory, and the rest of the data will have empty memory.

        :param tags: The tags of the memory allocation that will be loaded
            back to GPU memory. If None, all memory allocation will be loaded
            back to GPU memory.
        """
        successfully_mapped = []

        try:
            if self.use_cumem:
                # Use cumem-based wake up
                for ptr, data in self.pointer_to_data.items():
                    if tags is None or data.tag in tags:
                        handle = data.handle
                        if handle is None:
                            continue
                        create_and_map(handle)
                        successfully_mapped.append(ptr)

                        if data.cpu_backup_tensor is not None:
                            cpu_backup_tensor = data.cpu_backup_tensor
                            size_in_bytes = (
                                cpu_backup_tensor.numel()
                                * cpu_backup_tensor.element_size()
                            )
                            cpu_ptr = cpu_backup_tensor.data_ptr()
                            libcudart.cudaMemcpy(ptr, cpu_ptr, size_in_bytes)
                            data.cpu_backup_tensor = None
            else:
                # Fallback mode: restore tensors from CPU
                for tag, tensors in self.fallback_tensors.items():
                    if tags is not None and tag not in tags:
                        continue
                    for ptr, tensor in tensors.items():
                        data = self.pointer_to_data.get(ptr)
                        if data is None:
                            continue
                        if data.cpu_backup_tensor is not None:
                            # Restore from CPU backup
                            cpu_backup = data.cpu_backup_tensor
                            tensor.data = cpu_backup.to(tensor.device)
                            data.cpu_backup_tensor = None
                            successfully_mapped.append(ptr)

            self.is_sleeping = False

        except Exception as e:
            # Rollback on failure
            logger.error(
                "Failed to wake up allocator: %s. Rolling back %d successfully "
                "mapped allocations.",
                str(e),
                len(successfully_mapped),
            )
            if self.use_cumem:
                for ptr in successfully_mapped:
                    try:
                        data = self.pointer_to_data[ptr]
                        handle = data.handle
                        if handle:
                            unmap_and_release(handle)
                    except Exception as rollback_error:
                        logger.error(
                            "Failed to rollback allocation at ptr %s: %s",
                            ptr,
                            str(rollback_error),
                        )
            raise

    @contextmanager
    def use_memory_pool(self, tag: str | None = None):
        """
        A context manager to use the memory pool.
        All memory allocation created inside the context will be allocated
        in the memory pool, and has the specified tag.

        :param tag: The tag of the memory allocation. If None, the default tag
            will be used.
        """
        if tag is None:
            tag = CuMemAllocator.default_tag

        assert isinstance(tag, str)

        old_tag = self.current_tag
        self.current_tag = tag

        if self.use_cumem:
            with use_memory_pool_with_allocator(
                self.python_malloc_callback, self.python_free_callback
            ) as data:
                self.allocator_and_pools[tag] = data
                yield
                # Clean up unused allocations
                allocations = data[0].snapshot()
                for allocation in allocations:
                    if allocation["allocated_size"] == 0:
                        handle = self._python_free_callback(allocation["address"])
                        unmap_and_release(handle)
                self.current_tag = old_tag
        else:
            # Fallback mode: just track the tag
            if tag not in self.fallback_tensors:
                self.fallback_tensors[tag] = {}
            yield
            self.current_tag = old_tag

    def get_current_usage(self) -> int:
        """
        Get the total number of bytes allocated in the memory pool.
        """
        sum_bytes: int = 0
        for ptr, data in self.pointer_to_data.items():
            if self.use_cumem and data.handle:
                sum_bytes += data.handle[1]
            else:
                sum_bytes += data.size_bytes
        return sum_bytes


class FallbackMemoryManager:
    """
    A simpler memory manager for nano-vllm that doesn't require vLLM's C++ extensions.
    This provides sleep mode functionality using pure PyTorch operations.

    While not as efficient as cumem (doesn't truly release CUDA virtual memory),
    it still provides useful memory management for swapping model weights to CPU.
    """

    instance: "FallbackMemoryManager | None" = None

    @staticmethod
    def get_instance() -> "FallbackMemoryManager":
        if FallbackMemoryManager.instance is None:
            FallbackMemoryManager.instance = FallbackMemoryManager()
        return FallbackMemoryManager.instance

    def __init__(self):
        self.cpu_backups: dict[str, dict[str, torch.Tensor]] = {}
        self.is_sleeping: bool = False
        self.sleep_level: int = 0

    def sleep_model(
        self,
        model: torch.nn.Module,
        tag: str = "weights",
        level: int = 1,
    ) -> int:
        """
        Offload model weights to CPU and free GPU memory.

        Args:
            model: The model to sleep
            tag: Tag for this model's weights
            level: Sleep level (1 = offload weights, 2 = discard all)

        Returns:
            Number of bytes freed
        """
        if tag not in self.cpu_backups:
            self.cpu_backups[tag] = {}

        freed_bytes = 0

        for name, param in model.named_parameters():
            if param.device.type == "cuda":
                size_bytes = param.numel() * param.element_size()
                freed_bytes += size_bytes

                if level == 1:
                    # Level 1: Offload to CPU (pinned memory for faster transfer)
                    cpu_tensor = param.data.cpu()
                    if is_pin_memory_available():
                        cpu_tensor = cpu_tensor.pin_memory()
                    self.cpu_backups[tag][name] = cpu_tensor
                # Level 2: Just discard (no backup)

                # Clear the GPU tensor
                param.data = torch.empty(0, device=param.device, dtype=param.dtype)

        # Also handle buffers
        for name, buffer in model.named_buffers():
            if buffer.device.type == "cuda":
                size_bytes = buffer.numel() * buffer.element_size()
                freed_bytes += size_bytes

                if level == 1:
                    cpu_tensor = buffer.data.cpu()
                    if is_pin_memory_available():
                        cpu_tensor = cpu_tensor.pin_memory()
                    self.cpu_backups[tag][f"buffer_{name}"] = cpu_tensor

                buffer.data = torch.empty(0, device=buffer.device, dtype=buffer.dtype)

        self.is_sleeping = True
        self.sleep_level = level

        gc.collect()
        torch.cuda.empty_cache()

        return freed_bytes

    def wake_up_model(
        self,
        model: torch.nn.Module,
        tag: str = "weights",
        device: torch.device | str = "cuda",
    ) -> None:
        """
        Restore model weights from CPU backup.

        Args:
            model: The model to restore
            tag: Tag for this model's weights
            device: Device to restore to
        """
        if tag not in self.cpu_backups:
            logger.warning(f"No backup found for tag '{tag}'")
            return

        backups = self.cpu_backups[tag]

        for name, param in model.named_parameters():
            if name in backups:
                cpu_tensor = backups[name]
                param.data = cpu_tensor.to(device)

        for name, buffer in model.named_buffers():
            backup_name = f"buffer_{name}"
            if backup_name in backups:
                cpu_tensor = backups[backup_name]
                buffer.data = cpu_tensor.to(device)

        # Clear backups
        del self.cpu_backups[tag]

        self.is_sleeping = False
        self.sleep_level = 0

    def sleep_kv_cache(
        self,
        kv_cache: torch.Tensor,
        tag: str = "kv_cache",
    ) -> int:
        """
        Free KV cache memory (typically just discarded, not offloaded).

        Args:
            kv_cache: The KV cache tensor
            tag: Tag for this cache

        Returns:
            Number of bytes freed
        """
        if kv_cache is None:
            return 0

        freed_bytes = kv_cache.numel() * kv_cache.element_size()

        # KV cache is typically discarded, not offloaded
        # (it can be recomputed from input)
        kv_cache.data = torch.empty(0, device=kv_cache.device, dtype=kv_cache.dtype)

        gc.collect()
        torch.cuda.empty_cache()

        return freed_bytes
