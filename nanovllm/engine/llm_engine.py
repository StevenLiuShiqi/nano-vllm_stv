import atexit
import statistics
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
        self.step_latencies: list[float] = []
        self.num_prefill_steps = 0
        self.num_decode_steps = 0
        self.num_mixed_steps = 0
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
        t_step = perf_counter()
        prefill_seqs, decode_seqs = self.scheduler.schedule()

        # prefill (if any)
        prefill_token_ids = []
        if prefill_seqs:
            prefill_token_ids = self.model_runner.call("run", prefill_seqs, True)
            for seq in prefill_seqs:
                if self.scheduler.chunk_size is None:
                    seq.num_computed_tokens = seq.num_prompt_tokens
                else:
                    seq.num_computed_tokens = min(
                        seq.num_computed_tokens + self.scheduler.chunk_size,
                        seq.num_prompt_tokens
                    )

        # decode (if any)
        decode_token_ids = []
        if decode_seqs:
            decode_token_ids = self.model_runner.call("run", decode_seqs, False)

        self.scheduler.postprocess(prefill_seqs, prefill_token_ids)
        self.scheduler.postprocess(decode_seqs, decode_token_ids)

        all_seqs = prefill_seqs + decode_seqs
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in all_seqs if seq.is_finished] 

        if prefill_seqs:
            num_tokens = sum(len(seq) for seq in prefill_seqs)
        else:
            num_tokens = -len(decode_seqs)

        self.step_latencies.append(perf_counter() - t_step)
        if prefill_seqs and decode_seqs:
            self.num_mixed_steps += 1
        elif prefill_seqs:
            self.num_prefill_steps += 1
        else:
            self.num_decode_steps += 1

        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def clear_stats(self):
        self.step_latencies = []
        self.num_prefill_steps = 0
        self.num_decode_steps = 0
        self.num_mixed_steps = 0

    def get_stats(self) -> dict:
        lats_ms = [l * 1000 for l in self.step_latencies]
        if not lats_ms:
            return {"total_steps": 0}
        p99 = (statistics.quantiles(lats_ms, n=100)[98]
               if len(lats_ms) >= 100 else max(lats_ms))
        return {
            "total_steps": len(lats_ms),
            "prefill_steps": self.num_prefill_steps,
            "decode_steps": self.num_decode_steps,
            "mixed_steps": self.num_mixed_steps,
            "total_time_s": round(sum(lats_ms) / 1000, 3),
            "avg_step_ms": round(statistics.mean(lats_ms), 2),
            "p50_step_ms": round(statistics.median(lats_ms), 2),
            "p99_step_ms": round(p99, 2),
            "max_step_ms": round(max(lats_ms), 2),
        }

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
