#!/usr/bin/env python3
"""
Test harness: load one model with nano-vllm, sleep it, wait for user input, then wake it.
Useful to manually validate nano-vllm sleep/wake behavior.

Example:
    python test_sleep_model.py \
        --model /path/to/model \
        --level 1 \
        --max-tokens 8

Options let you tweak TP size and gpu_memory_utilization.

Sleep Mode Explanation:
    Sleep mode uses CUDA virtual memory management to truly release GPU memory
    while keeping the ability to restore the model state. This is different from
    simply moving tensors to CPU with .to("cpu"), which doesn't release the
    underlying CUDA memory allocations.

    - Level 1: Offload weights to CPU, discard KV cache (can quickly restore)
    - Level 2: Discard everything (need to reload from disk)
"""
from __future__ import annotations

import argparse
import sys
from time import perf_counter

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sleep one model, wait for input, then wake it (nano-vllm)")
    p.add_argument("--model", required=True, help="Path to model")
    p.add_argument("--level", type=int, default=1, choices=[1, 2],
                   help="Sleep level (1=offload weights, 2=discard all)")
    p.add_argument("--max-tokens", type=int, default=8, help="Tokens for tiny gen")
    p.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    p.add_argument("--gpu-mem", type=float, default=0.9, help="GPU mem util")
    p.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graphs")
    p.add_argument(
        "--enable-sleep-mode",
        default=True,
        action="store_true",
        help="Enable nano-vllm sleep mode memory pool",
    )
    return p.parse_args()


def get_gpu_memory_info() -> tuple[float, float, float]:
    """Get GPU memory info in GiB."""
    free, total = torch.cuda.mem_get_info()
    used = total - free
    return free / 1024**3, used / 1024**3, total / 1024**3


def print_gpu_memory():
    """Print current GPU memory usage."""
    free, used, total = get_gpu_memory_info()
    print(f"GPU Memory: {used:.2f} GiB used / {total:.2f} GiB total ({free:.2f} GiB free)")


def make_llm(path: str, tp: int, gpu_mem: float, enforce_eager: bool,
             enable_sleep_mode: bool):
    """Create a nano-vllm LLM instance."""
    # Add nano-vllm to path if needed
    sys.path.insert(0, ".")

    from nanovllm import LLM
    return LLM(
        model=path,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_mem,
        enforce_eager=enforce_eager,
        enable_sleep_mode=enable_sleep_mode,
    )


def tiny_gen(llm, text: str, max_tokens: int) -> str:
    """Generate a small amount of text."""
    from nanovllm import SamplingParams
    sp = SamplingParams(max_tokens=max_tokens)
    out = llm.generate([text], sp, use_tqdm=False)
    return out[0]["text"] if isinstance(out[0], dict) else out[0].outputs[0].text


def check_cumem_extension() -> bool:
    """Check if cumem_allocator C++ extension is available."""
    try:
        from nanovllm.device_allocator.cumem import cumem_available
        return cumem_available
    except ImportError:
        return False


def main() -> int:
    args = parse_args()

    # Check cumem extension availability
    cumem_available = check_cumem_extension()

    print("=" * 60)
    print("nano-vllm Sleep Mode Test")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Sleep level: {args.level}")
    print(f"Enable sleep mode: {args.enable_sleep_mode}")
    print(f"Tensor parallel size: {args.tp}")
    print(f"GPU memory utilization: {args.gpu_mem}")
    print(f"Enforce eager: {args.enforce_eager}")
    print(f"cumem_allocator extension: {'AVAILABLE' if cumem_available else 'NOT AVAILABLE (using fallback)'}")
    if not cumem_available:
        print("\n*** NOTE: For proper GPU memory release, compile the C++ extension:")
        print("    cd nano-vllm && pip install -e .")
    print("=" * 60)

    print("\n== Initial GPU Memory ==")
    print_gpu_memory()

    print("\n== Loading Model ==")
    t0 = perf_counter()
    llm = make_llm(
        args.model,
        args.tp,
        args.gpu_mem,
        args.enforce_eager,
        args.enable_sleep_mode,
    )
    t1 = perf_counter()
    print(f"Model loaded in {(t1 - t0) * 1000:.1f} ms")
    print_gpu_memory()

    # Initial generation test
    print("\n== Initial Generation Test ==")
    try:
        txt = tiny_gen(llm, "Hello, how are you?", args.max_tokens)
        print(f"Generated: {txt!r}")
    except Exception as e:
        print(f"Initial generation failed: {e}")
        return 1

    # Wait for user to sleep
    while True:
        user_input = input("\n模型已加载，输入 y 后 sleep: ").strip().lower()
        if user_input == "y":
            break
        print("请输入 y 继续。")

    # Sleep the model
    print(f"\n== Sleeping Model (level={args.level}) ==")
    print("Before sleep:")
    print_gpu_memory()

    t2 = perf_counter()
    stats = llm.sleep(level=args.level)
    t3 = perf_counter()

    print(f"Sleep completed in {(t3 - t2) * 1000:.1f} ms")
    print(f"Freed: {stats.get('freed_bytes', 0) / 1024**3:.2f} GiB")
    print("After sleep:")
    print_gpu_memory()

    # Verify model is sleeping
    print(f"\nModel is sleeping: {llm.is_sleeping()}")

    # Wait for user to wake
    while True:
        user_input = input("\n模型已sleep，输入 y 后 wake_up: ").strip().lower()
        if user_input == "y":
            break
        print("请输入 y 继续。")

    # Wake up the model
    print("\n== Waking Up Model ==")
    print("Before wake_up:")
    print_gpu_memory()

    t4 = perf_counter()
    llm.wake_up()
    t5 = perf_counter()

    print(f"Wake up completed in {(t5 - t4) * 1000:.1f} ms")
    print("After wake_up:")
    print_gpu_memory()

    # Verify model is awake
    print(f"\nModel is sleeping: {llm.is_sleeping()}")

    # Wait for user to test generation
    while True:
        user_input = input("\n模型已wake，输入 y 后进行tiny gen: ").strip().lower()
        if user_input == "y":
            break
        print("请输入 y 继续。")

    # Post-wake generation test
    print("\n== Post-Wake Generation Test ==")
    try:
        txt = tiny_gen(llm, "Hello again after waking up!", args.max_tokens)
        print(f"Generated: {txt!r}")
        print("\nSleep/Wake cycle completed successfully!")
    except Exception as e:
        print(f"Post-wake generation failed: {e}")
        return 1

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  - Model load time: {(t1 - t0) * 1000:.1f} ms")
    print(f"  - Sleep time (level {args.level}): {(t3 - t2) * 1000:.1f} ms")
    print(f"  - Wake up time: {(t5 - t4) * 1000:.1f} ms")
    print(f"  - Memory freed during sleep: {stats.get('freed_bytes', 0) / 1024**3:.2f} GiB")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
