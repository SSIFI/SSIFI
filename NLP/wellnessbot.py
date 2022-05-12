import os, platform
import torch.nn as nn
import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from NLP.kogpt_wellness import DialogKoGPT2

ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

if platform.system() == "Windows":
	base_path = str(os.path.abspath(__file__)).split('\\')
else:
	base_path = str(os.path.abspath(__file__)).split('/')
base_path = '/'.join(base_path[:-1])
tokenizer_path = base_path + '/models/wellnessbot/tokenizer'
model_path = base_path + '/models/wellnessbot/model'
checkpoint_path = base_path + '/models/wellnessbot/checkpoint/wellnessbot_last.pth'

# Wellnessbot
def wellnessbot(prompt, max_length = 50):
	# Default max length : 50

	checkpoint = torch.load(checkpoint_path, map_location=device)
	model = DialogKoGPT2()
	model.load_state_dict(checkpoint['model_state_dict'], strict=False)
	tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
	with torch.no_grad():
		tokens = tokenizer.encode(prompt)

		input_ids = torch.tensor([tokenizer.bos_token_id,]  + tokens +[tokenizer.eos_token_id]).unsqueeze(0)
		generated = model.generate(input_ids=input_ids)
		generated = tokenizer.decode(generated[0].tolist()[len(tokens)+1:],skip_special_tokens=True)

	# "." 에서 문장 종료
	if "." in generated:
		endpoint = generated.index(".")
		generated = generated[:endpoint+1]
	return generated