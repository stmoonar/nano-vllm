import gc
import logging
import pickle
import torch
import torch.distributed as dist
from contextlib import nullcontext
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model

logger = logging.getLogger(__name__)


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.enable_sleep_mode = config.enable_sleep_mode

        # Sleep mode state
        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}
        self._is_sleeping = False
        self._sleep_level = 0

        # CUDA graph state (will be populated by capture_cudagraph if not enforce_eager)
        self.graphs = None
        self.graph_pool = None
        self.graph_vars = None
        self.graph_bs = None

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")

        # Use memory pool context if sleep mode is enabled
        with self._maybe_get_memory_pool_context("weights"):
            self.model = Qwen3ForCausalLM(hf_config)
            load_model(self.model, config.model)

        self.sampler = Sampler()
        self.warmup_model()

        with self._maybe_get_memory_pool_context("kv_cache"):
            self.allocate_kv_cache()

        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self, realloc: bool = False):
        config = self.config
        hf_config = config.hf_config
        device = f"cuda:{self.rank}"

        free, total = torch.cuda.mem_get_info()
        used = total - free
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize

        if realloc and config.num_kvcache_blocks > 0:
            # Use previously calculated number of blocks for reallocation
            num_blocks = config.num_kvcache_blocks
        else:
            # Calculate based on memory stats
            peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
            current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
            num_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
            config.num_kvcache_blocks = num_blocks

        assert num_blocks > 0, f"Not enough GPU memory for KV cache. Free: {free / 1024**3:.2f} GiB"

        self.kv_cache = torch.empty(
            2, hf_config.num_hidden_layers, num_blocks, self.block_size, num_kv_heads, head_dim,
            device=device, dtype=hf_config.torch_dtype
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        # Use eager mode if CUDA graphs not available (e.g., during sleep)
        use_eager = self.enforce_eager or self.graphs is None or input_ids.size(0) > 512
        if is_prefill or use_eager:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        device = f"cuda:{self.rank}"
        input_ids = torch.zeros(max_bs, dtype=torch.int64, device=device)
        positions = torch.zeros(max_bs, dtype=torch.int64, device=device)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32, device=device)
        context_lens = torch.zeros(max_bs, dtype=torch.int32, device=device)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=device)
        outputs = torch.zeros(max_bs, hf_config.hidden_size, device=device)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

    def _maybe_get_memory_pool_context(self, tag: str):
        """
        Get a memory pool context for the specified tag if sleep mode is enabled.

        Args:
            tag: The tag for the memory allocation ("weights" or "kv_cache")

        Returns:
            A context manager for the memory pool, or nullcontext if sleep mode is disabled
        """
        if self.enable_sleep_mode:
            from nanovllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            if tag == "weights":
                # Ensure memory pool is empty for weights allocation
                if allocator.get_current_usage() > 0:
                    logger.warning(
                        "Sleep mode memory pool is not empty. "
                        "This may indicate multiple instances sharing the same pool."
                    )
            return allocator.use_memory_pool(tag=tag)
        else:
            return nullcontext()

    def sleep(self, level: int = 1) -> dict:
        """
        Put the model in sleep mode to free GPU memory.

        Args:
            level: Sleep level
                - Level 1: Offload model weights to CPU, discard KV cache
                - Level 2: Discard everything (weights and KV cache)

        Returns:
            Dictionary with memory statistics
        """
        from nanovllm.device_allocator.cumem import cumem_available

        free_bytes_before = torch.cuda.mem_get_info()[0]

        # Release CUDA graphs first (they hold references to GPU memory)
        if not self.enforce_eager and self.graphs is not None:
            del self.graphs
            del self.graph_pool
            del self.graph_vars
            self.graphs = None
            self.graph_pool = None
            self.graph_vars = None
            gc.collect()

        # Save buffers before level 2 sleep (they need to be restored)
        if level == 2:
            self._sleep_saved_buffers = {
                name: buffer.cpu().clone()
                for name, buffer in self.model.named_buffers()
            }

        if self.enable_sleep_mode and cumem_available:
            # Use CuMemAllocator (vLLM's CUDA virtual memory API)
            from nanovllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            # Level 1: offload weights, discard kv_cache
            # Level 2: discard all (no offload)
            allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())
        else:
            # Fallback: use model.to("cpu")
            from nanovllm.device_allocator.cumem import FallbackMemoryManager

            mem_manager = FallbackMemoryManager.get_instance()
            mem_manager.sleep_model(self.model, tag="weights", level=level)

            # For fallback, manually clear KV cache
            if hasattr(self, 'kv_cache') and self.kv_cache is not None:
                del self.kv_cache
                self.kv_cache = None

        # Force garbage collection and cache clearing
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        free_bytes_after, total = torch.cuda.mem_get_info()
        freed_bytes = free_bytes_after - free_bytes_before
        used_bytes = total - free_bytes_after

        self._is_sleeping = True
        self._sleep_level = level

        logger.info(
            "Sleep mode (level %d) freed %.2f GiB memory, %.2f GiB still in use.",
            level,
            freed_bytes / 1024**3,
            used_bytes / 1024**3,
        )

        return {
            "freed_bytes": freed_bytes,
            "used_bytes": used_bytes,
            "level": level,
        }

    def wake_up(self, tags: list[str] | None = None) -> None:
        """
        Wake up the model from sleep mode.

        Args:
            tags: Tags to wake up. If None, wake up all tags.
                  Valid tags: "weights", "kv_cache"
        """
        from nanovllm.device_allocator.cumem import cumem_available

        if not self._is_sleeping:
            logger.warning("Model is not sleeping, nothing to wake up.")
            return

        if self.enable_sleep_mode and cumem_available:
            # Use CuMemAllocator
            from nanovllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            allocator.wake_up(tags)
        else:
            # Fallback
            from nanovllm.device_allocator.cumem import FallbackMemoryManager

            mem_manager = FallbackMemoryManager.get_instance()
            if tags is None or "weights" in tags:
                mem_manager.wake_up_model(self.model, tag="weights", device=f"cuda:{self.rank}")

        # Restore saved buffers after level 2 sleep
        if self._sleep_saved_buffers:
            for name, buffer in self.model.named_buffers():
                if name in self._sleep_saved_buffers:
                    saved = self._sleep_saved_buffers[name]
                    buffer.data = saved.to(f"cuda:{self.rank}")
            self._sleep_saved_buffers = {}

        # Re-allocate KV cache (kv_cache is always discarded during sleep)
        if tags is None or "kv_cache" in tags:
            with self._maybe_get_memory_pool_context("kv_cache"):
                self.allocate_kv_cache(realloc=True)

        # Re-capture CUDA graphs
        if not self.enforce_eager:
            self.capture_cudagraph()

        self._is_sleeping = False
        self._sleep_level = 0

        logger.info("Model woke up from sleep mode.")

    def is_sleeping(self) -> bool:
        """Check if the model is currently in sleep mode."""
        return self._is_sleeping
