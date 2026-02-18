# SPDX-License-Identifier: Apache-2.0
"""
cumem-based pytorch pluggable allocator to implement sleep mode for nano-vllm.

This module directly uses vLLM's cumem allocator implementation for proper
CUDA virtual memory management. The cumem allocator uses CUDA driver APIs
to truly release GPU memory, unlike simple tensor.to("cpu") operations.

Key concepts:
- Uses PyTorch pluggable allocator with CUDA virtual memory APIs
- Tensors created in use_memory_pool() context are tracked by tag
- sleep() unmaps GPU memory and optionally backs up to CPU
- wake_up() remaps GPU memory and restores from CPU backup
"""
import dataclasses
import gc
import logging
import os
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch

logger = logging.getLogger(__name__)


# Try to import vLLM's cumem allocator
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
    logger.info("nano-vllm: Using vLLM's cumem allocator for sleep mode")
except (ModuleNotFoundError, ImportError) as e:
    init_module = None
    python_create_and_map = None
    python_unmap_and_release = None
    CudaRTLibrary = None
    lib_name = None
    libcudart = None
    logger.warning(f"nano-vllm: vLLM cumem not available ({e}), sleep mode will be limited")


def is_pin_memory_available() -> bool:
    """Check if pinned memory is available."""
    return torch.cuda.is_available()


# py_device, py_alignedSize, py_d_mem, py_p_memHandle
HandleType = tuple[int, int, int, int]


@dataclasses.dataclass
class AllocationData:
    """Data structure to track memory allocations."""
    handle: HandleType
    tag: str
    cpu_backup_tensor: torch.Tensor | None = None


def create_and_map(allocation_handle: HandleType) -> None:
    """Create and map GPU memory using CUDA virtual memory API."""
    python_create_and_map(*allocation_handle)


def unmap_and_release(allocation_handle: HandleType) -> None:
    """Unmap and release GPU memory using CUDA virtual memory API."""
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

    This is a direct port of vLLM's CuMemAllocator for nano-vllm.

    Inside the `use_memory_pool(tag)` context, all tensors created will
    be allocated in the memory pool, and has the same tag as the
    tag passed to the context.

    When we call `sleep`, all tensors with the specified tag will be
    offloaded to CPU memory, and the rest of the tensors will be discarded.
    When we call `wake_up`, all tensors that are previously offloaded
    will be loaded back to GPU memory.
    """

    instance: "CuMemAllocator | None" = None
    default_tag: str = "default"

    @staticmethod
    def get_instance() -> "CuMemAllocator":
        """Get the singleton instance."""
        assert cumem_available, "cumem allocator is not available"
        if CuMemAllocator.instance is None:
            CuMemAllocator.instance = CuMemAllocator()
        return CuMemAllocator.instance

    def __init__(self):
        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        assert "expandable_segments:True" not in conf, (
            "Expandable segments are not compatible with memory pool. "
            "Please track https://github.com/pytorch/pytorch/issues/147851 "
            "for the latest updates."
        )

        self.pointer_to_data: dict[int, AllocationData] = {}
        self.current_tag: str = CuMemAllocator.default_tag
        self.allocator_and_pools: dict[str, Any] = {}
        # Creating strong references to prevent garbage collection
        self.python_malloc_callback = self._python_malloc_callback
        self.python_free_callback = self._python_free_callback

    def _python_malloc_callback(self, allocation_handle: HandleType) -> None:
        """Store allocation data when memory is allocated."""
        py_d_mem = allocation_handle[2]
        self.pointer_to_data[py_d_mem] = AllocationData(
            allocation_handle, self.current_tag
        )
        logger.debug(
            "Allocated %s bytes for %s with address %s",
            allocation_handle[1], self.current_tag, py_d_mem,
        )

    def _python_free_callback(self, ptr: int) -> HandleType:
        """Look up allocation data when memory is freed."""
        data = self.pointer_to_data.pop(ptr)
        if data.cpu_backup_tensor is not None:
            data.cpu_backup_tensor = None
        logger.debug(
            "Freed %s bytes for %s with address %s",
            data.handle[1], data.tag, ptr,
        )
        return data.handle

    def sleep(self, offload_tags: tuple[str, ...] | str | None = None) -> None:
        """
        Put the allocator in sleep mode.

        All data with the specified tag will be offloaded to CPU memory,
        and others will be discarded.

        :param offload_tags: Tags to offload. Others will be discarded.
        """
        if offload_tags is None:
            offload_tags = (CuMemAllocator.default_tag,)
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags,)

        assert isinstance(offload_tags, tuple)

        total_bytes = 0
        backup_bytes = 0

        for ptr, data in self.pointer_to_data.items():
            handle = data.handle
            total_bytes += handle[1]
            if data.tag in offload_tags:
                # Backup to CPU using cudaMemcpy
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
            # Unmap and release GPU memory
            unmap_and_release(handle)

        logger.info(
            "CuMemAllocator: sleep freed %.2f GiB memory in total, of which "
            "%.2f GiB is backed up in CPU and the rest %.2f GiB is discarded.",
            total_bytes / 1024**3,
            backup_bytes / 1024**3,
            (total_bytes - backup_bytes) / 1024**3,
        )

        gc.collect()
        torch.cuda.empty_cache()

    def wake_up(self, tags: list[str] | None = None) -> None:
        """
        Wake up the allocator from sleep mode.

        All data that was previously offloaded will be loaded back to GPU.

        :param tags: Tags to wake up. If None, all tags are woken up.
        """
        successfully_mapped = []

        try:
            for ptr, data in self.pointer_to_data.items():
                if tags is None or data.tag in tags:
                    handle = data.handle
                    # Remap GPU memory
                    create_and_map(handle)
                    successfully_mapped.append(ptr)

                    # Restore from CPU backup
                    if data.cpu_backup_tensor is not None:
                        cpu_backup_tensor = data.cpu_backup_tensor
                        size_in_bytes = (
                            cpu_backup_tensor.numel()
                            * cpu_backup_tensor.element_size()
                        )
                        cpu_ptr = cpu_backup_tensor.data_ptr()
                        libcudart.cudaMemcpy(ptr, cpu_ptr, size_in_bytes)
                        data.cpu_backup_tensor = None
        except Exception as e:
            # Rollback on failure
            logger.error(
                "Failed to wake up allocator: %s. Rolling back %d allocations.",
                str(e), len(successfully_mapped),
            )
            for ptr in successfully_mapped:
                try:
                    data = self.pointer_to_data[ptr]
                    unmap_and_release(data.handle)
                except Exception as rollback_error:
                    logger.error("Failed to rollback at ptr %s: %s", ptr, rollback_error)
            raise

    @contextmanager
    def use_memory_pool(self, tag: str | None = None):
        """
        Context manager to use the memory pool.

        All tensors created inside this context will be allocated in the
        memory pool with the specified tag.
        """
        if tag is None:
            tag = CuMemAllocator.default_tag

        assert isinstance(tag, str)

        old_tag = self.current_tag
        self.current_tag = tag
        with use_memory_pool_with_allocator(
            self.python_malloc_callback, self.python_free_callback
        ) as data:
            self.allocator_and_pools[tag] = data
            yield
            # Clean up unused allocations (PyTorch bug workaround)
            allocations = data[0].snapshot()
            for allocation in allocations:
                if allocation["allocated_size"] == 0:
                    handle = self._python_free_callback(allocation["address"])
                    unmap_and_release(handle)
            self.current_tag = old_tag

    def get_current_usage(self) -> int:
        """Get total bytes allocated in the memory pool."""
        return sum(data.handle[1] for data in self.pointer_to_data.values())


class FallbackMemoryManager:
    """
    Fallback memory manager when vLLM's cumem is not available.

    This uses model.to("cpu") which is less efficient but still works.
    Note: This doesn't truly release CUDA virtual memory like cumem does.
    """

    instance: "FallbackMemoryManager | None" = None

    @staticmethod
    def get_instance() -> "FallbackMemoryManager":
        if FallbackMemoryManager.instance is None:
            FallbackMemoryManager.instance = FallbackMemoryManager()
        return FallbackMemoryManager.instance

    def __init__(self):
        self.sleep_info: dict[str, dict] = {}
        self.is_sleeping: bool = False
        self.sleep_level: int = 0

    def sleep_model(
        self,
        model: torch.nn.Module,
        tag: str = "weights",
        level: int = 1,
    ) -> int:
        """Offload model to CPU."""
        freed_bytes = 0
        for param in model.parameters():
            if param.device.type == "cuda":
                freed_bytes += param.numel() * param.element_size()
        for buffer in model.buffers():
            if buffer.device.type == "cuda":
                freed_bytes += buffer.numel() * buffer.element_size()

        if level == 1:
            model.to("cpu")
            self.sleep_info[tag] = {"level": level, "device": "cuda"}
        else:
            # Level 2: weights will be lost
            for param in model.parameters():
                if param.device.type == "cuda":
                    param.data = torch.empty(param.shape, device="cpu", dtype=param.dtype)
            for buffer in model.buffers():
                if buffer.device.type == "cuda":
                    buffer.data = torch.empty(buffer.shape, device="cpu", dtype=buffer.dtype)
            self.sleep_info[tag] = {"level": level, "device": "cuda"}

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
        """Restore model to GPU."""
        if tag not in self.sleep_info:
            logger.warning(f"No sleep info found for tag '{tag}'")
            return

        model.to(device)
        del self.sleep_info[tag]

        if not self.sleep_info:
            self.is_sleeping = False
            self.sleep_level = 0
