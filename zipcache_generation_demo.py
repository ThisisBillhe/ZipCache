import torch
from transformers import AutoTokenizer

from zipcache import MyLlamaForCausalLM
compress_config = {}
## Key compress config
compress_config = {}
compress_config["compress_mode"] = "mixed_channelwiseQ"
compress_config["quantize_bit_important"] = 4
compress_config["quantize_bit_unimportant"] = 2
compress_config["k_unimportant_ratio"] = 0.4
## Value compress config
compress_config["v_compress_mode"] = "channel_separate_mixed_tokenwiseQ"
compress_config["v_quantize_bit_important"] = 4
compress_config["v_quantize_bit_unimportant"] = 2
compress_config["v_unimportant_ratio"] = 0.4
compress_config["stream"] = True # streaming-gear set to true to perform better efficiency
compress_config["streaming_gap"] = 100 # re-compress every N iteration

MODEL_PATH='/data/models--meta-llama--Meta-Llama-3-8B/snapshots/1460c22666392e470910ce3d44ffeb2ab7dbd4df/' ## your llama path here

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, use_fast=True, cache_dir=MODEL_PATH, local_files_only=True
)
with open('asset/gsm8k_sample.txt', 'r') as file:
    prompt_text = file.read()
input_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors='pt').input_ids.cuda()

if 'Llama' in MODEL_PATH:
    model = MyLlamaForCausalLM.from_pretrained(
        MODEL_PATH,
        cache_dir=MODEL_PATH,
        compress_config=compress_config,
        torch_dtype=torch.float16,
        local_files_only=True
    )
else:
    raise NotImplementedError

model.half().eval().cuda()

generate_kwargs = dict(
    return_dict_in_generate=False,
    max_new_tokens=128,
    output_scores=False,
    pad_token_id=tokenizer.eos_token_id,
    use_cache=True,
)

generate_kwargs["do_sample"] = False
generate_kwargs["temperature"] = None
generate_kwargs["top_k"] = None
generate_kwargs["top_p"] = None

generate_ids = model.generate(input_ids, **generate_kwargs)
result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("################## Generated Context with Our Cache ###################")
print(result)
