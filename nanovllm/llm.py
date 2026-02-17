from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """
    The main LLM class for nano-vllm.

    This class extends LLMEngine to provide a high-level API for LLM inference,
    including sleep mode support for efficient GPU memory management.

    Example:
        >>> from nanovllm import LLM, SamplingParams
        >>> llm = LLM(model="/path/to/model", enable_sleep_mode=True)
        >>> outputs = llm.generate(["Hello, world!"], SamplingParams())
        >>> llm.sleep(level=1)  # Offload weights to CPU
        >>> # ... do something else with GPU memory ...
        >>> llm.wake_up()  # Restore weights to GPU
        >>> outputs = llm.generate(["Hello again!"], SamplingParams())
    """

    def sleep(self, level: int = 1) -> dict:
        """
        Put the model in sleep mode to free GPU memory.

        Sleep mode uses CUDA virtual memory management (via cumem) to truly release
        GPU memory while keeping the ability to restore the model state. This is
        different from simply moving tensors to CPU with .to("cpu"), which doesn't
        release the underlying CUDA memory allocations.

        Supported sleep levels:
            - Level 1: Offload model weights to CPU memory, discard KV cache.
                       The weights can be quickly restored from CPU.
            - Level 2: Discard all GPU memory (weights and KV cache).
                       The model needs to be reloaded from disk on wake_up.

        Args:
            level: Sleep level (1 or 2). Default is 1.

        Returns:
            Dictionary with memory statistics:
                - freed_bytes: Number of bytes freed
                - used_bytes: Number of bytes still in use
                - level: The sleep level used

        Example:
            >>> llm = LLM(model="/path/to/model", enable_sleep_mode=True)
            >>> # Do some inference...
            >>> stats = llm.sleep(level=1)
            >>> print(f"Freed {stats['freed_bytes'] / 1e9:.2f} GB")
        """
        return super().sleep(level)

    def wake_up(self, tags: list[str] | None = None) -> None:
        """
        Wake up the model from sleep mode.

        This restores the model weights from CPU memory (for level 1 sleep) or
        triggers a reload from disk (for level 2 sleep). The KV cache will be
        re-allocated as needed.

        Args:
            tags: Optional list of tags to wake up. If None, all tags are woken up.
                  Valid tags: "weights", "kv_cache"

        Example:
            >>> llm.sleep(level=1)
            >>> # ... free GPU memory used for something else ...
            >>> llm.wake_up()  # Restore everything
            >>> # Or wake up only weights first:
            >>> llm.wake_up(tags=["weights"])
        """
        super().wake_up(tags)

    def is_sleeping(self) -> bool:
        """
        Check if the model is currently in sleep mode.

        Returns:
            True if the model is sleeping, False otherwise.

        Example:
            >>> llm.sleep(level=1)
            >>> assert llm.is_sleeping() == True
            >>> llm.wake_up()
            >>> assert llm.is_sleeping() == False
        """
        return super().is_sleeping()
