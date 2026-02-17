import atexit
import logging
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs

    def sleep(self, level: int = 1) -> dict:
        """
        Put the model in sleep mode to free GPU memory.

        This allows temporarily freeing GPU memory while keeping the model state,
        useful for scenarios like:
        - Running multiple models on the same GPU
        - RLHF workflows where reward models are loaded/unloaded
        - Memory-constrained inference scenarios

        Args:
            level: Sleep level
                - Level 1: Offload model weights to CPU, discard KV cache
                - Level 2: Discard everything (weights and KV cache)

        Returns:
            Dictionary with memory statistics including freed_bytes and used_bytes
        """
        # Clear scheduler state
        self.scheduler.clear()

        # Call sleep on model_runner
        result = self.model_runner.call("sleep", level)

        logger.info(
            "LLMEngine: Model entered sleep mode (level=%d). "
            "Freed %.2f GiB, %.2f GiB still in use.",
            level,
            result.get("freed_bytes", 0) / 1024**3,
            result.get("used_bytes", 0) / 1024**3,
        )

        return result

    def wake_up(self, tags: list[str] | None = None) -> None:
        """
        Wake up the model from sleep mode.

        Args:
            tags: Tags to wake up. If None, wake up all tags.
                  Valid tags: "weights", "kv_cache"
        """
        t0 = perf_counter()

        # Call wake_up on model_runner
        self.model_runner.call("wake_up", tags)

        t1 = perf_counter()
        logger.info(
            "LLMEngine: Model woke up from sleep mode in %.2f ms.",
            (t1 - t0) * 1000,
        )

    def is_sleeping(self) -> bool:
        """Check if the model is currently in sleep mode."""
        return self.model_runner._is_sleeping
