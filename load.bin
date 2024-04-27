from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, AutoConfig
import torch

model = LlamaForCausalLM.from_pretrained(
    'tinyllama',
    load_in_8bit=False,
    torch_dtype=torch.float32,
    token="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",
)
