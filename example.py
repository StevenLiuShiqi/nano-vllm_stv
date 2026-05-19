import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def run_and_report(label: str, chunk_size, prompts, sampling_params, model_path):
    print(f"\n{'='*60}\n{label}  (chunk_size={chunk_size})\n{'='*60}")
    kwargs = {"enforce_eager": True, "tensor_parallel_size": 1}
    if chunk_size is not None:
        kwargs["chunk_size"] = chunk_size
    llm = LLM(model_path, **kwargs)

    blkmngr = llm.scheduler.block_manager
    llm.clear_stats()
    blkmngr.clear_stats()

    outputs = llm.generate(prompts, sampling_params)

    print("engine stats:", llm.get_stats())
    print("block_manager stats:", blkmngr.get_stats())
    for prompt, output in zip(prompts, outputs):
        print(f"Prompt: {prompt[:80]}")
        print(f"Completion: {output['text'][:100]}")
    return llm.get_stats()


def main():
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

    baseline = run_and_report("Baseline (no chunking)", None, prompts, sampling_params, path)
    chunked = run_and_report("Chunked prefill", 128, prompts, sampling_params, path)

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
