import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForCausalLM, GPT2LMHeadModel, PreTrainedTokenizerFast

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)

save_dir_tokenizer = ''
save_dir_model = ''

has_cuda = torch.cuda.is_available()
ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

# BASIC MODEL
save_dir_tokenizer = './models/basicmodel/tokenizer'
save_dir_model = './models/basicmodel/model'
Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 
model = AutoModelWithLMHead.from_pretrained('skt/kogpt2-base-v2')
tokenizer.save_pretrained(save_dir_tokenizer)
model.save_pretrained(save_dir_model)

# NOVEL BOT
save_dir_tokenizer = './models/novelbot/tokenizer'
save_dir_model = './models/novelbot/model'
tokenizer = AutoTokenizer.from_pretrained("ttop324/kogpt2novel")
model = AutoModelWithLMHead.from_pretrained("ttop324/kogpt2novel")
tokenizer.save_pretrained(save_dir_tokenizer)
model.save_pretrained(save_dir_model)

# WELLNESS BOT
save_dir_tokenizer = './models/wellnessbot/tokenizer'
save_dir_model = './models/wellnessbot/model'
tokenizer = PreTrainedTokenizerFast.from_pretrained("taeminlee/kogpt2")
model = GPT2LMHeadModel.from_pretrained("taeminlee/kogpt2")
tokenizer.save_pretrained(save_dir_tokenizer)
model.save_pretrained(save_dir_model)

# PAINTER BOT
save_dir_diffusion = './models/painterbot/diffusion'
save_dir_model = './models/painterbot/model'
options = model_and_diffusion_defaults()
options['use_fp16'] = has_cuda
options['timestep_respacing'] = '100' # use 100 diffusion steps for fast sampling
model, diffusion = create_model_and_diffusion(**options)
model.eval()
if has_cuda:
    model.convert_to_fp16()
model.to(device)
model.load_state_dict(load_checkpoint('base', device))

os.makedirs('./models/painterbot', exist_ok=True)

torch.save(model, save_dir_model)
torch.save(diffusion, save_dir_diffusion)
print('total upsampler parameters', sum(x.numel() for x in model.parameters()))


# kogpt - kakaobrain
save_dir_tokenizer = './models/kogpt_kakao/tokenizer'
save_dir_model = './models/kogpt_kakao/model'
tokenizer = AutoTokenizer.from_pretrained(
    'kakaobrain/kogpt', revision = 'KoGPT6B-ryan1.5b-float16',
    bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Current device:', device)
model = AutoModelForCausalLM.from_pretrained(
    'kakaobrain/kogpt', revision = 'KoGPT6B-ryan1.5b-float16',
    pad_token_id=tokenizer.eos_token_id,
    torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device=device, non_blocking=True)
_=model.eval()
tokenizer.save_pretrained(save_dir_tokenizer)
model.save_pretrained(save_dir_model)