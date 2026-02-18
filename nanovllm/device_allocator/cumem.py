# SPDX-License-Identifier: Apache-2.0
"""
CUDA memory management for nano-vllm sleep mode.

This module implements sleep mode functionality using CUDA virtual memory APIs
via the cumem_allocator C++ extension. This approach truly releases GPU memory
by unmapping virtual memory, unlike simple .to("cpu") which doesn't release
the underlying CUDA memory allocations.

Sleep Mode Levels:
- Level 1: Offload weights to CPU pinned memory, can quickly restore
- Level 2: Discard weights entirely (must reload from disk)
"""
import ctypes
import dataclasses
import gc
import logging
import os
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def is_pin_memory_available() -> bool:
    """Check if pinned memory is available."""
    return torch.cuda.is_available()


def find_loaded_library(lib_name: str) -> str | None:
    """Find a loaded library by name."""
    import ctypes.util

    # Try to find the library
    lib_path = ctypes.util.find_library(lib_name)
    if lib_path:
        return lib_path

    # Try common paths on Linux
    if os.name != 'nt':
        common_paths = [
            f'/usr/lib/{lib_name}.so',
            f'/usr/lib64/{lib_name}.so',
            f'/usr/local/lib/{lib_name}.so',
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path

    return None


# ---------------------------------------------------------------------------
# CudaRTLibrary: Pure Python wrapper for cudart library
# ---------------------------------------------------------------------------

cudaError_t = ctypes.c_int
cudaMemcpyKind = ctypes.c_int


@dataclasses.dataclass
class Function:
    name: str
    restype: Any
    argtypes: list[Any]


class CudaRTLibrary:
    """Pure Python wrapper for the cudart library using ctypes."""

    exported_functions = [
        Function("cudaSetDevice", cudaError_t, [ctypes.c_int]),
        Function("cudaDeviceSynchronize", cudaError_t, []),
        Function("cudaGetErrorString", ctypes.c_char_p, [cudaError_t]),
        Function(
            "cudaMalloc",
            cudaError_t,
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
        ),
        Function("cudaFree", cudaError_t, [ctypes.c_void_p]),
        Function(
            "cudaMemcpy",
            cudaError_t,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, cudaMemcpyKind],
        ),
    ]

    # ROCm function name mapping
    cuda_to_hip_mapping = {
        "cudaSetDevice": "hipSetDevice",
        "cudaDeviceSynchronize": "hipDeviceSynchronize",
        "cudaGetErrorString": "hipGetErrorString",
        "cudaMalloc": "hipMalloc",
        "cudaFree": "hipFree",
        "cudaMemcpy": "hipMemcpy",
    }

    # Cache for loaded libraries
    path_to_library_cache: dict[str, Any] = {}
    path_to_dict_mapping: dict[str, dict[str, Any]] = {}

    def __init__(self, so_file: str | None = None):
        if so_file is None:
            so_file = self._find_cudart_library()

        if so_file is None:
            raise RuntimeError(
                "libcudart not found. Please ensure CUDA is installed and "
                "set CUDA_HOME or CUDA_PATH environment variable."
            )

        if so_file not in CudaRTLibrary.path_to_library_cache:
            lib = ctypes.CDLL(so_file)
            CudaRTLibrary.path_to_library_cache[so_file] = lib
        self.lib = CudaRTLibrary.path_to_library_cache[so_file]

        if so_file not in CudaRTLibrary.path_to_dict_mapping:
            _funcs = {}
            is_rocm = 'hip' in so_file.lower() or 'rocm' in so_file.lower()
            for func in CudaRTLibrary.exported_functions:
                func_name = (
                    CudaRTLibrary.cuda_to_hip_mapping[func.name]
                    if is_rocm
                    else func.name
                )
                try:
                    f = getattr(self.lib, func_name)
                    f.restype = func.restype
                    f.argtypes = func.argtypes
                    _funcs[func.name] = f
                except AttributeError:
                    logger.warning(f"Function {func_name} not found in {so_file}")
            CudaRTLibrary.path_to_dict_mapping[so_file] = _funcs
        self.funcs = CudaRTLibrary.path_to_dict_mapping[so_file]

    def _find_cudart_library(self) -> str | None:
        """Find the cudart library."""
        # Import torch to ensure CUDA libraries are loaded
        import torch  # noqa

        # Try to find libcudart
        lib_path = find_loaded_library("cudart")
        if lib_path:
            return lib_path

        # Try environment variables
        cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
        if cuda_home:
            if os.name == 'nt':
                candidate = os.path.join(cuda_home, 'bin', 'cudart64_12.dll')
                if os.path.exists(candidate):
                    return candidate
                # Try other versions
                for ver in ['12', '11', '110', '111', '112', '120', '121']:
                    candidate = os.path.join(cuda_home, 'bin', f'cudart64_{ver}.dll')
                    if os.path.exists(candidate):
                        return candidate
            else:
                candidate = os.path.join(cuda_home, 'lib64', 'libcudart.so')
                if os.path.exists(candidate):
                    return candidate

        # Try common Linux paths
        if os.name != 'nt':
            common_paths = [
                '/usr/local/cuda/lib64/libcudart.so',
                '/usr/lib/x86_64-linux-gnu/libcudart.so',
            ]
            for path in common_paths:
                if os.path.exists(path):
                    return path

        return None

    def CUDART_CHECK(self, result: cudaError_t) -> None:
        if result != 0:
            error_str = self.cudaGetErrorString(result)
            raise RuntimeError(f"CUDART error: {error_str}")

    def cudaGetErrorString(self, error: cudaError_t) -> str:
        return self.funcs["cudaGetErrorString"](error).decode("utf-8")

    def cudaSetDevice(self, device: int) -> None:
        self.CUDART_CHECK(self.funcs["cudaSetDevice"](device))

    def cudaDeviceSynchronize(self) -> None:
        self.CUDART_CHECK(self.funcs["cudaDeviceSynchronize"]())

    def cudaMemcpy(
        self, dst: ctypes.c_void_p, src: ctypes.c_void_p, count: int
    ) -> None:
        cudaMemcpyDefault = 4
        kind = cudaMemcpyDefault
        self.CUDART_CHECK(self.funcs["cudaMemcpy"](dst, src, count, kind))


# ---------------------------------------------------------------------------
# Try to import the C++ extension
# ---------------------------------------------------------------------------

cumem_available = False
init_module = None
python_create_and_map = None
python_unmap_and_release = None
lib_name = None
libcudart = None

try:
    from nanovllm.cumem_allocator import (
        init_module,
        python_create_and_map,
        python_unmap_and_release,
    )

    # Find the compiled extension library
    import nanovllm.cumem_allocator as _cumem_module
    lib_name = _cumem_module.__file__

    # Initialize cudart library
    libcudart = CudaRTLibrary()

    cumem_available = True
    logger.info("nano-vllm: cumem_allocator extension loaded successfully")
except ImportError as e:
    logger.warning(
        f"nano-vllm: cumem_allocator extension not available ({e}). "
        "Sleep mode will use fallback implementation."
    )


# py_device, py_alignedSize, py_d_mem, py_p_memHandle
HandleType = tuple[int, int, int, int]


@dataclasses.dataclass
class AllocationData:
    handle: HandleType
    tag: str
    cpu_backup_tensor: torch.Tensor | None = None


def create_and_map(allocation_handle: HandleType) -> None:
    python_create_and_map(*allocation_handle)


def unmap_and_release(allocation_handle: HandleType) -> None:
    python_unmap_and_release(*allocation_handle)


def get_pluggable_allocator(
    python_malloc_fn: Callable[[int], int],
    python_free_func: Callable[[int, int], None]
) -> torch.cuda.memory.CUDAPluggableAllocator:
    init_module(python_malloc_fn, python_free_func)
    new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
        lib_name, "my_malloc", "my_free"
    )
    return new_alloc


@contextmanager
def use_memory_pool_with_allocator(
    python_malloc_fn: Callable[[int], int],
    python_free_func: Callable[[int, int], None]
):
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
        # Creating strong references to the two callbacks here to prevent
        # these ephemeral bound-method objects being garbage collected.
        self.python_malloc_callback = self._python_malloc_callback
        self.python_free_callback = self._python_free_callback

    def _python_malloc_callback(self, allocation_handle: HandleType) -> None:
        """
        Internal method to store the allocation data
        when memory is allocated in the memory pool."""
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
        return

    def _python_free_callback(self, ptr: int) -> HandleType:
        """
        Internal method to look up the allocation data
        when memory is freed in the memory pool."""
        data = self.pointer_to_data.pop(ptr)
        if data.cpu_backup_tensor is not None:
            data.cpu_backup_tensor = None
        logger.debug(
            "Freed %s bytes for %s with address %s from cumem allocator",
            data.handle[1],
            data.tag,
            ptr,
        )
        return data.handle

    def sleep(self, offload_tags: tuple[str, ...] | str | None = None) -> int:
        """
        Put the allocator in sleep mode.
        All data in the memory allocation with the specified tag will be
        offloaded to CPU memory, and others will be discarded.

        :param offload_tags: The tags of the memory allocation that will be
            offloaded. The rest of the memory allocation will be discarded.
        :return: Total bytes freed
        """
        if offload_tags is None:
            # by default, allocated tensors are offloaded
            # when the allocator sleeps
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

        return total_bytes

    def wake_up(self, tags: list[str] | None = None) -> None:
        """
        Wake up the allocator from sleep mode.
        All data that is previously offloaded will be loaded back to GPU
        memory, and the rest of the data will have empty memory.

        :param tags: The tags of the memory allocation that will be loaded
            back to GPU memory. If None, all memory allocation will be loaded
            back to GPU memory.
        """
        # Track successfully mapped allocations for rollback on failure
        successfully_mapped = []

        try:
            for ptr, data in self.pointer_to_data.items():
                if tags is None or data.tag in tags:
                    handle = data.handle
                    create_and_map(handle)
                    successfully_mapped.append(ptr)

                    if data.cpu_backup_tensor is not None:
                        cpu_backup_tensor = data.cpu_backup_tensor
                        if cpu_backup_tensor is not None:
                            size_in_bytes = (
                                cpu_backup_tensor.numel()
                                * cpu_backup_tensor.element_size()
                            )
                            cpu_ptr = cpu_backup_tensor.data_ptr()
                            libcudart.cudaMemcpy(ptr, cpu_ptr, size_in_bytes)
                            data.cpu_backup_tensor = None
        except Exception as e:
            # Rollback all successfully mapped allocations on failure
            logger.error(
                "Failed to wake up allocator: %s. Rolling back %d successfully "
                "mapped allocations.",
                str(e),
                len(successfully_mapped),
            )
            for ptr in successfully_mapped:
                try:
                    data = self.pointer_to_data[ptr]
                    handle = data.handle
                    unmap_and_release(handle)
                except Exception as rollback_error:
                    logger.error(
                        "Failed to rollback allocation at ptr %s: %s",
                        ptr,
                        str(rollback_error),
                    )
            # Re-raise the original exception after cleanup
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
        with use_memory_pool_with_allocator(
            self.python_malloc_callback, self.python_free_callback
        ) as data:
            self.allocator_and_pools[tag] = data
            yield
            # Find all unused allocations and manually release them.
            allocations = data[0].snapshot()
            for allocation in allocations:
                if allocation["allocated_size"] == 0:
                    handle = self._python_free_callback(allocation["address"])
                    unmap_and_release(handle)
            self.current_tag = old_tag

    def get_current_usage(self) -> int:
        """
        Get the total number of bytes allocated in the memory pool.
        """
        sum_bytes: int = 0
        for ptr, data in self.pointer_to_data.items():
            handle = data.handle
            sum_bytes += handle[1]
        return sum_bytes


# ---------------------------------------------------------------------------
# Fallback implementation when C++ extension is not available
# ---------------------------------------------------------------------------

class FallbackSleepModeManager:
    """
    Fallback manager for model sleep mode when C++ extension is not available.

    This uses simple tensor operations which don't truly release GPU memory
    but can still be used for testing purposes.
    """

    instance: "FallbackSleepModeManager | None" = None

    @staticmethod
    def get_instance() -> "FallbackSleepModeManager":
        if FallbackSleepModeManager.instance is None:
            FallbackSleepModeManager.instance = FallbackSleepModeManager()
        return FallbackSleepModeManager.instance

    def __init__(self):
        # Store CPU backups: {tag: {param_name: (cpu_tensor, original_shape, original_dtype, original_device)}}
        self.cpu_backups: dict[str, dict[str, tuple[torch.Tensor, torch.Size, torch.dtype, torch.device]]] = {}
        self.is_sleeping: bool = False
        self.sleep_level: int = 0

    def sleep_model(
        self,
        model: nn.Module,
        tag: str = "weights",
        level: int = 1,
    ) -> int:
        """
        Put model into sleep mode.

        Note: This fallback implementation doesn't truly release GPU memory.
        For proper memory release, compile and use the C++ extension.
        """
        if tag not in self.cpu_backups:
            self.cpu_backups[tag] = {}

        freed_bytes = 0

        # Backup and release parameters
        for name, param in model.named_parameters():
            if param.device.type == "cuda":
                size_bytes = param.numel() * param.element_size()
                freed_bytes += size_bytes

                if level == 1:
                    # Level 1: Backup to CPU
                    cpu_backup = param.data.to("cpu", non_blocking=False)
                    self.cpu_backups[tag][f"param_{name}"] = (
                        cpu_backup, param.shape, param.dtype, param.device
                    )

                # Clear the tensor
                param.data = torch.empty(0, dtype=param.dtype, device=param.device)

        # Backup and release buffers
        for name, buffer in model.named_buffers():
            if buffer.device.type == "cuda":
                size_bytes = buffer.numel() * buffer.element_size()
                freed_bytes += size_bytes

                if level == 1:
                    cpu_backup = buffer.data.to("cpu", non_blocking=False)
                    self.cpu_backups[tag][f"buffer_{name}"] = (
                        cpu_backup, buffer.shape, buffer.dtype, buffer.device
                    )

                buffer.data = torch.empty(0, dtype=buffer.dtype, device=buffer.device)

        self.is_sleeping = True
        self.sleep_level = level

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        logger.warning(
            "FallbackSleepModeManager: Using fallback implementation. "
            "GPU memory may not be fully released. "
            "Reported freed: %.2f GiB (level=%d, tag=%s)",
            freed_bytes / 1024**3, level, tag,
        )

        return freed_bytes

    def wake_up_model(
        self,
        model: nn.Module,
        tag: str = "weights",
        device: str | torch.device = "cuda",
    ) -> None:
        """Wake up model from sleep mode."""
        if tag not in self.cpu_backups:
            logger.warning(f"No backup found for tag '{tag}'")
            return

        backups = self.cpu_backups[tag]

        # Restore parameters
        for name, param in model.named_parameters():
            backup_name = f"param_{name}"
            if backup_name in backups:
                cpu_tensor, shape, dtype, orig_device = backups[backup_name]
                # Ensure we restore with the correct dtype and device
                param.data = cpu_tensor.to(device=device, dtype=dtype)

        # Restore buffers
        for name, buffer in model.named_buffers():
            backup_name = f"buffer_{name}"
            if backup_name in backups:
                cpu_tensor, shape, dtype, orig_device = backups[backup_name]
                # Ensure we restore with the correct dtype and device
                buffer.data = cpu_tensor.to(device=device, dtype=dtype)

        # Clear backup
        del self.cpu_backups[tag]

        if not self.cpu_backups:
            self.is_sleeping = False
            self.sleep_level = 0

        logger.info("FallbackSleepModeManager: Restored model to GPU (tag=%s)", tag)


# For backward compatibility
SleepModeManager = FallbackSleepModeManager
