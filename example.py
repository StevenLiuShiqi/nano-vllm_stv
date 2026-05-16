import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("/root/autodl-tmp/huggingface/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=2048)
    prompts = [                                                                 
        "Please analyze the following text and provide a detailed summary: " + "The quick brown fox jumps over the lazy dog. " * 30,                       
        "introduce yourself",
    ]  
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    attn = llm.model_runner.model.model.layers[0].self_attn.attn

    attn.clear_stats()
    outputs1 = llm.generate(prompts[:2], sampling_params)
    print(attn.get_stats())
    attn.clear_stats()

    outputs2 = llm.generate(prompts[:2], sampling_params)
    attn = llm.model_runner.model.model.layers[0].self_attn.attn            
    print(attn.get_stats())

    # outputs = llm.generate(prompts, sampling_params)
    # attn = llm.model_runner.model.model.layers[0].self_attn.attn 
    # print(attn.get_stats()) 

    for prompt, output in zip(prompts[:2], outputs1):
      print(f"Prompt: {prompt[:100]}")                                            
      print(f"Completion: {output['text']}") 
                                                                              
    for prompt, output in zip(prompts[:2], outputs2):                           
      print(f"Prompt: {prompt[:100]}")             
      print(f"Completion: {output['text']}") 


if __name__ == "__main__":
    main()
