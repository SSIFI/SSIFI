import os
import platform

if platform.system() == "Windows":
    base_path = str(os.path.abspath(__file__)).split('\\')
else:
    base_path = str(os.path.abspath(__file__)).split('/')
base_path = '/'.join(base_path[:-1])

### kss ###
dataset = "kss"
data_path = os.path.join(base_path + "/dataset/", dataset)
meta_name = "transcript.v.1.4.txt"
textgrid_name = "TextGrid.zip"

### set GPU number ###
train_visible_devices = "0,1"
synth_visible_devices = "0"

# Text
text_cleaners = ['korean_cleaners']

# Audio and mel
### kss ###
sampling_rate = 22050
filter_length = 1024
hop_length = 256
win_length = 1024

### kss ###
max_wav_value = 32768.0
n_mel_channels = 80
mel_fmin = 0
mel_fmax = 8000

f0_min = 71.0
f0_max = 792.8
energy_min = 0.0
energy_max = 283.72


# FastSpeech 2
encoder_layer = 4
encoder_head = 2
encoder_hidden = 256
decoder_layer = 4
decoder_head = 2
decoder_hidden = 256
fft_conv1d_filter_size = 1024
fft_conv1d_kernel_size = (9, 1)
encoder_dropout = 0.2
decoder_dropout = 0.2

variance_predictor_filter_size = 256
variance_predictor_kernel_size = 3
variance_predictor_dropout = 0.5

max_seq_len = 1000

# Checkpoints and synthesis path
preprocessed_path = os.path.join(base_path + "/preprocessed/", dataset)
checkpoint_path = os.path.join(base_path + "/ckpt/", dataset)
eval_path = os.path.join(base_path + "/eval/", dataset)
log_path = os.path.join(base_path + "/log/", dataset)
# test_path = str(os.getcwd())+"/results"


# Optimizer
batch_size = 16
epochs = 1000
n_warm_up_step = 4000
grad_clip_thresh = 1.0
acc_steps = 1

betas = (0.9, 0.98)
eps = 1e-9
weight_decay = 0.


# Vocoder
vocoder = 'vocgan'
vocoder_pretrained_model_name = "vocgan_kss_pretrained_model_epoch_4500.pt"
vocoder_pretrained_model_path = os.path.join(
    base_path + "/vocoder/pretrained_models/", vocoder_pretrained_model_name)

# Log-scaled duration
log_offset = 1.


# Save, log and synthesis
save_step = 10000
eval_step = 1000
eval_size = 256
log_step = 1000
clear_Time = 20
