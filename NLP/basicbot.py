import os
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

base_path = str(os.path.abspath(__file__)).split('\\')
base_path = '/'.join(base_path[:-1])
tokenizer_path = base_path + '/models/basicbot/tokenizer'
model_path = base_path + '/models/basicbot/model'

# BASIC BOT
def basicbot(prompt, max_length: int = 256):
  tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
  model = GPT2LMHeadModel.from_pretrained(model_path)
  with torch.no_grad():
    tokens = tokenizer.encode(prompt, return_tensors='pt')
    gen_tokens = model.generate(tokens, do_sample=True, temperature=0.8, max_length=max_length, reqpetition_penalty=2.0, use_cache=True)
    generated = tokenizer.batch_decode(gen_tokens)[0]
  return generated