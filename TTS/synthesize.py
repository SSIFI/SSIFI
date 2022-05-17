from jamo import h2j
from g2pk import G2p
import codecs
from . import audio as Audio
from . import utils
from .text import text_to_sequence, sequence_to_text
from .vocoder import vocgan_generator
from .fastspeech2 import FastSpeech2
from string import punctuation
import re
import argparse
import torch
import torch.nn as nn
import numpy as np
from . import hparams as hp
import os
import platform

if platform.system() == "Windows":
    base_path = str(os.path.abspath(__file__)).split('\\')
else:
    base_path = str(os.path.abspath(__file__)).split('/')
base_path = '/'.join(base_path[:-1])

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = hp.synth_visible_devices


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def kor_preprocess(text):
    text = text.rstrip(punctuation)

    g2p = G2p()
    phone = g2p(text)
    print('after g2p: ', phone)
    phone = h2j(phone)
    print('after h2j: ', phone)
    phone = list(filter(lambda p: p != ' ', phone))
    phone = '{' + '}{'.join(phone) + '}'
    print('phone: ', phone)
    phone = re.sub(r'\{[^\w\s]?\}', '{sp}', phone)
    print('after re.sub: ', phone)
    phone = phone.replace('}{', ' ')

    print('|' + phone + '|')
    sequence = np.array(text_to_sequence(phone, hp.text_cleaners))
    sequence = np.stack([sequence])
    return torch.from_numpy(sequence).long().to(device)


def get_FastSpeech2(num):
    checkpoint_path = os.path.join(
        hp.checkpoint_path, "checkpoint_{}.pth.tar".format(num))
    model = nn.DataParallel(FastSpeech2())
    model.load_state_dict(torch.load(
        checkpoint_path, map_location=device)['model'])
    model.requires_grad = False
    model.eval()
    return model


def synthesize(file_name, results_path, model, vocoder, text):

    mean_mel, std_mel = torch.tensor(np.load(os.path.join(
        hp.preprocessed_path, "mel_stat.npy")), dtype=torch.float).to(device)
    mean_f0, std_f0 = torch.tensor(np.load(os.path.join(
        hp.preprocessed_path, "f0_stat.npy")), dtype=torch.float).to(device)
    mean_energy, std_energy = torch.tensor(np.load(os.path.join(
        hp.preprocessed_path, "energy_stat.npy")), dtype=torch.float).to(device)

    mean_mel, std_mel = mean_mel.reshape(1, -1), std_mel.reshape(1, -1)
    mean_f0, std_f0 = mean_f0.reshape(1, -1), std_f0.reshape(1, -1)
    mean_energy, std_energy = mean_energy.reshape(
        1, -1), std_energy.reshape(1, -1)

    src_len = torch.from_numpy(np.array([text.shape[1]])).to(device)

    mel, mel_postnet, log_duration_output, f0_output, energy_output, _, _, mel_len = model(
        text, src_len)

    mel_torch = mel.transpose(1, 2).detach()
    mel_postnet_torch = mel_postnet.transpose(1, 2).detach()
    f0_output = f0_output[0]
    energy_output = energy_output[0]

    mel_torch = utils.de_norm(mel_torch.transpose(1, 2), mean_mel, std_mel)
    mel_postnet_torch = utils.de_norm(mel_postnet_torch.transpose(
        1, 2), mean_mel, std_mel).transpose(1, 2)
    f0_output = utils.de_norm(
        f0_output, mean_f0, std_f0).squeeze().detach().cpu().numpy()
    energy_output = utils.de_norm(
        energy_output, mean_energy, std_energy).squeeze().detach().cpu().numpy()

    if vocoder is not None:
        # VocGAN으로 음성 파일 생성
        if hp.vocoder.lower() == "vocgan":
            utils.vocgan_infer(mel_postnet_torch, vocoder, path=os.path.join(
                results_path, '{}.wav'.format(file_name)))
    else:
        # griffin_lim으로 음성 파일 생성
        Audio.tools.inv_mel_spec(mel_postnet_torch[0], os.path.join(
            hp.results_path, '{}.wav'.format(file_name)))

    utils.plot_data([(mel_postnet_torch[0].detach().cpu().numpy(), f0_output, energy_output)], [
                    'Synthesized Spectrogram'], filename=os.path.join(results_path, '{}.png'.format(file_name)))


def make_sound(file_name, sentence, results_path, step=350000):

    model = get_FastSpeech2(step).to(device)

    if hp.vocoder == 'vocgan':
        vocoder = utils.get_vocgan(ckpt_path=hp.vocoder_pretrained_model_path)
    else:
        vocoder = None

    vocoder = utils.get_vocgan(ckpt_path=hp.vocoder_pretrained_model_path)

    text = kor_preprocess(sentence)

    synthesize(file_name, results_path, model, vocoder, text)
