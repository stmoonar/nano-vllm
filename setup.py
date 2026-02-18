"""
Setup script for nano-vllm with CUDA extension support.

This script compiles the cumem_allocator extension which provides
CUDA virtual memory management for sleep mode functionality.
"""
import os
import sys
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class CUDAExtension(Extension):
    """Custom extension class for CUDA extensions."""
    pass


class BuildExt(build_ext):
    """Custom build_ext command that handles CUDA compilation."""

    def build_extensions(self):
        # Check if CUDA is available
        cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')

        if cuda_home is None:
            # Try common locations
            common_paths = [
                '/usr/local/cuda',
                '/usr/local/cuda-12',
                '/usr/local/cuda-11',
                'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0',
                'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8',
            ]
            for path in common_paths:
                if os.path.exists(path):
                    cuda_home = path
                    break

        if cuda_home is None:
            print("Warning: CUDA not found. Skipping cumem_allocator extension.")
            print("Sleep mode will use fallback implementation.")
            return

        print(f"Found CUDA at: {cuda_home}")

        # Set up compiler flags
        for ext in self.extensions:
            if isinstance(ext, CUDAExtension):
                # Add CUDA include directories
                ext.include_dirs.append(os.path.join(cuda_home, 'include'))

                # Add CUDA library directories
                if sys.platform == 'win32':
                    ext.library_dirs.append(os.path.join(cuda_home, 'lib', 'x64'))
                else:
                    ext.library_dirs.append(os.path.join(cuda_home, 'lib64'))

                # Link against CUDA libraries
                ext.libraries.extend(['cuda', 'cudart'])

                # Set compiler flags
                if self.compiler.compiler_type == 'msvc':
                    ext.extra_compile_args = ['/O2', '/std:c++17']
                else:
                    ext.extra_compile_args = ['-O3', '-std=c++17', '-fPIC']
                    ext.extra_link_args = ['-fPIC']

        build_ext.build_extensions(self)


def get_extensions():
    """Get the list of extensions to build."""
    extensions = []

    # cumem_allocator extension
    cumem_ext = CUDAExtension(
        'nanovllm.cumem_allocator',
        sources=['csrc/cumem_allocator.cpp'],
        include_dirs=['csrc'],
        language='c++',
    )
    extensions.append(cumem_ext)

    return extensions


# Check if we should skip extension building
skip_ext = os.environ.get('NANOVLLM_SKIP_EXT', '0') == '1'

setup(
    name='nano-vllm',
    version='0.2.0',
    author='Xingkai Yu',
    description='a lightweight vLLM implementation built from scratch',
    packages=find_packages(include=['nanovllm*']),
    python_requires='>=3.10,<3.13',
    install_requires=[
        'torch>=2.4.0',
        'triton>=3.0.0',
        'transformers>=4.51.0',
        'flash-attn',
        'xxhash',
    ],
    ext_modules=[] if skip_ext else get_extensions(),
    cmdclass={'build_ext': BuildExt} if not skip_ext else {},
)
