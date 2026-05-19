import argparse
import json
import os
import subprocess
import sys


def run_once(chunk_size):
    import os
    from nanovllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    path = os.path.expanduser("/root/autodl-tmp/huggingface/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(path)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=2048)
    raw_prompts = [
        "Please analyze the following text and provide a detailed summary: "
        + "The quick brown fox jumps over the lazy dog. " * 30,
        "introduce yourself",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in raw_prompts
    ]

    kwargs = {"enforce_eager": True, "tensor_parallel_size": 1}
    if chunk_size is not None:
        kwargs["chunk_size"] = chunk_size
    llm = LLM(path, **kwargs)

    blkmngr = llm.scheduler.block_manager
    llm.clear_stats()
    blkmngr.clear_stats()

    outputs = llm.generate(prompts, sampling_params)

    stats = llm.get_stats()
    print("__STATS__" + json.dumps(stats))
    print("block_manager stats:", blkmngr.get_stats())
    for prompt, output in zip(prompts, outputs):
        print(f"Prompt: {prompt[:80]}")
        print(f"Completion: {output['text'][:100]}")


def run_subprocess(label, chunk_size):
    print(f"\n{'='*60}\n{label}  (chunk_size={chunk_size})\n{'='*60}")
    cmd = [sys.executable, __file__, "--child"]
    if chunk_size is not None:
        cmd += ["--chunk-size", str(chunk_size)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stats = None
    for line in result.stdout.splitlines():
        if line.startswith("__STATS__"):
            stats = json.loads(line[len("__STATS__"):])
        else:
            print(line)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
    return stats or {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=None)
    args = parser.parse_args()

    if args.child:
        run_once(args.chunk_size)
        return

    baseline = run_subprocess("Baseline (no chunking)", None)
    chunked = run_subprocess("Chunked prefill", 128)

    print(f"\n{'='*60}\nComparison\n{'='*60}")
    keys = ["total_steps", "prefill_steps", "decode_steps", "mixed_steps",
            "total_time_s", "avg_step_ms", "p50_step_ms", "p99_step_ms", "max_step_ms"]
    print(f"{'metric':<18}{'baseline':>14}{'chunked':>14}")
    for k in keys:
        b = baseline.get(k, "-")
        c = chunked.get(k, "-")
        print(f"{k:<18}{str(b):>14}{str(c):>14}")


if __name__ == "__main__":
    main()
