import torch
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForCausalLM

# kogpt - kakao brain

def kogpt(prompt, max_length: int = 256):
  tokenizer_path = './models/kogpt_kakao/tokenizer'
  model_path = './models/kogpt_kakao/model'
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print('Current device:', device)
  model = AutoModelForCausalLM.from_pretrained(
      model_path,
      pad_token_id=tokenizer.eos_token_id,
      torch_dtype=torch.float16, low_cpu_mem_usage=True
      ).to(device=device, non_blocking=True)
  _=model.eval()
  with torch.no_grad():
    tokens = tokenizer.encode(prompt, return_tensors='pt').to(device=device, non_blocking=True)
    gen_tokens = model.generate(tokens, do_sample=True, temperature=0.8, max_length=max_length)
    generated = tokenizer.batch_decode(gen_tokens)[0]
  return generated