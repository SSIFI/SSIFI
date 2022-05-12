import os, platform
import torch.nn as nn
from kogpt2_transformers import get_kogpt2_model

if platform.system() == "Windows":
  base_path = str(os.path.abspath(__file__)).split('\\')
else:
  base_path = str(os.path.abspath(__file__)).split('/')
base_path = '/'.join(base_path[:-1])
tokenizer_path = base_path + '/models/wellnessbot/tokenizer'
model_path = base_path + '/models/wellnessbot/model'

class DialogKoGPT2(nn.Module):
  def __init__(self):
    super(DialogKoGPT2, self).__init__()
    self.kogpt2 = get_kogpt2_model(model_path)

  def generate(self,
               input_ids,
               do_sample=True,
               max_length= 60,
               top_p=0.92,
               top_k=50,
               temperature= 0.6,
               no_repeat_ngram_size =None,
               num_return_sequences=3,
               early_stopping=False,
               ):
    return self.kogpt2.generate(input_ids,
               do_sample=do_sample,
               max_length=max_length,
               top_p = top_p,
               top_k=top_k,
               temperature=temperature,
               no_repeat_ngram_size= no_repeat_ngram_size,
               num_return_sequences=num_return_sequences,
               early_stopping = early_stopping,
              )

  def forward(self, input, labels = None):
    if labels is not None:
      outputs = self.kogpt2(input, labels=labels)
    else:
      outputs = self.kogpt2(input)

    return outputs