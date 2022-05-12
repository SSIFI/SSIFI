

# SSIFI

## Intro

>  이 프로젝트의 목표는 대화형 AI 대한 접근성을 높이는 것 입니다. 이 오픈소스를 통해서 사용자는 STT, NLP, TTS 기술을 보다 쉽게 사용하실 수 있습니다.



## Service

(프로젝트 서비스 이미지)

- 해당 서비스는 오픈소스를 바탕으로 제작한 서비스 입니다.



## Install Dependencies

먼저 python = 3.7, [pytorch](https://pytorch.org/), [ffmpeg](https://ffmpeg.org/)와 [g2pk](https://github.com/Kyubyong/g2pK)를 설치합니다.
```
# ffmpeg install
sudo apt-get install ffmpeg

# [WARNING] g2pk를 설치하시기 전에, g2pk github을 참조하셔서 g2pk의 dependency를 설치하시고 g2pk를 설치하시기 바랍니다.
pip install g2pk
```

해당 설치 과정은 TTS에서 사용되는 패키지의 의존성 문제 때문에 선행으로 진행하게 됩니다.



다음으로, 필요한 모듈을 pip를 이용하여 설치합니다.

```
pip install -r requirements.txt
```

**[WARNING]**
**1. Window에서는 g2pk의 dependency인 python-mecab-ko 설치문제가 발생하며, Window 버전을 설치 하더라도 g2pk 설치시 에러가 발생할 수 있기 때문에 linux 환경을 권장드립니다.**

**2. Anaconda 가상환경을 사용하시는 것을 권장드립니다.**



## How to use

먼저 ssifi.py에서 STT, NLP, TTS를 사용하기 위한 변수값을 지정한 후 아래의 커맨드를 실행해 주세요.

```
python ssifi.py
```



# STT(Sound To Text) - Speech Recognition(Korean)

> 본 프로젝트에서는 STT로 [Speech Recognition](https://github.com/Uberi/speech_recognition) 라이브러리를 사용했습니다. 해당 라이브러리는 온라인 및 오프라인에서 여러 엔진 및 API를 지원하여 음성 인식을 수행하기 위한 라이브러리 입니다. 라이브러리에는 다양한 기능이 있지만, 현재 프로젝트에서는 한국어 음성이 녹음된 파일을 이용하기 때문에 한국어 음성을 텍스트화 하는 기능만 사용합니다.

- Speech recognition engine/API 
  - [CMU Sphinx](http://cmusphinx.sourceforge.net/wiki/) (works offline)
  - Google Speech Recognition(프로젝에서 사용한 모델.)
  - [Google Cloud Speech API](https://cloud.google.com/speech/)
  - [Wit.ai](https://wit.ai/)
  - [Microsoft Azure Speech](https://azure.microsoft.com/en-us/services/cognitive-services/speech/)
  - [Microsoft Bing Voice Recognition (Deprecated)](https://www.microsoft.com/cognitive-services/en-us/speech-api)
  - [Houndify API](https://houndify.com/)
  - [IBM Speech to Text](http://www.ibm.com/smarterplanet/us/en/ibmwatson/developercloud/speech-to-text.html)
  - [Snowboy Hotword Detection](https://snowboy.kitt.ai/) (works offline)
  - [Tensorflow](https://www.tensorflow.org/)
  - [Vosk API](https://github.com/alphacep/vosk-api/) (works offline)



## Transcription

```python
# STT
audio_path = './STT/audio_sample/korean.wav'
prompt = STT.speech_recognition(audio_path)
```

- ssifi.py 
  - audio_path: 텍스트화 하고자 하는 음성파일의 경로
  - prompt: 텍스트화 결과값(NLP의 input) 



# NLP(Natural Language Processing)



## pre-trained model



## Fine-Tuning



# TTS(Text To Sound) - FastSpeech 2 Korean

> 이 프로젝트의 TTS는 Microsoft의 [**FastSpeech 2(Y. Ren et. al., 2020)**](https://arxiv.org/abs/2006.04558)를 [**Korean Single Speech dataset (이하 KSS dataset)**](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset)에서 동작하도록 구현한 것입니다. 
>
> 본 소스코드는 https://github.com/HGU-DLLAB/Korean-FastSpeech2-Pytorch 를 변형 및 수정한 것입니다.
>
> 변경 사항
> - preprocess.py, train.py, synthesize.py의 실행 부분을 함수화
> - 모듈 사용을 위한 경로 관련 코드 추가 및 수정
> - ssifi.py를 통해 음성 생성하기 위해 필요한 변수 추가
>   -  생성된 음성파일 저장경로(result_path)
>   - 파일명(file_name)
>   - 트레이닝 된 모델의 step



## Preprocessing

**(1) kss dataset download**

* [Korean-Single-Speech dataset](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset): 12,853개(약 12시간)의 샘플로 구성된 한국어 여성 단일화자 발화 dataset입니다.

dataset을 다운로드 하신 후, 압축을 해제하시고 ``hparams.py``에 있는 ``data_path``에 다운받은 kss dataset의 경로를 기록해주세요.

**(2) phoneme-utterance sequence간 alignment 정보 download**

* KSS ver.1.4. ([download](https://drive.google.com/file/d/1LgZPfWAvPcdOpGBSncvMgv54rGIf1y-H/view?usp=sharing))

FastSpeech2를 학습하기 위해서는 [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/)(MFA)에서 추출된 utterances와 phoneme sequence간의 alignment가 필요합니다. kss dataset에 대한 alignment 정보(TextGrid)는 위의 링크에서 다운로드 가능합니다. 다운 받은 ```TextGrid.zip```파일을 ``프로젝트 폴더 (TTS)``에 두시면 됩니다. 

***KSS dataset에 적용된 License로 인해 kss dataset에서 추출된 TextGrid를 상업적으로 사용하는 것을 금합니다.**

**(3) 데이터 전처리**

1. hparms.py
- dataset : dataset의 폴더 이름
- data_path : dataset의 상위 폴더 경로
- meta_name : metadata의 text 파일명 ex)transcript.v.1.4.txt
- tetxgrid_name : textgird 압푹 파일의 파일명
2. preprocess.py
- line 42:`` if "kss" in hp.dataset:`` 에서 kss 부분은 본인이 설정한 dataset 이름을 작성하시면 됩니다.

```
python TTS_preprocess.py
```
data 전처리를 위해 위의 커맨드를 입력해 주세요. 전처리 된 데이터는 프로젝트 폴더의 ``TTS/preprocessed/`` 폴더에 생성됩니다.



## Training
모델 학습 전에, kss dataset에 대해 사전학습된 VocGAN(neural vocoder)을 [다운로드](https://drive.google.com/file/d/1GxaLlTrEhq0aXFvd_X1f4b-ev7-FH8RB/view?usp=sharing) 하여 ``vocoder/pretrained_models/`` 경로에 위치시킵니다.
아래의 커맨드를 입력하여 모델 학습을 수행합니다.

```
python TTS_train.py
```
학습된 모델은 ``ckpt/``에 저장되고 tensorboard log는 ``log/``에 저장됩니다. 학습시 evaluate 과정에서 생성된 음성은 ``eval/`` 폴더에 저장됩니다.

만약, 학습된 모델에 이어서 하시려면 TTS_train.py의 ``start_train``함수에서 변수 값으로 학습된 모델의 step 값을 넣어주시면 됩니다.



## Pre-trained Model
pretrained model(checkpoint)을 [다운로드](https://drive.google.com/file/d/1qkFuNLqPIm-A5mZZDPGK1mnp0_Lh00PN/view?usp=sharing)해 주세요.
그 후,  ```hparams.py```에 있는 ```checkpoint_path``` 변수에 기록된 경로에 위치시켜주시면 사전학습된 모델을 사용 가능합니다.



## Synthesis

학습된 파라미터를 기반으로 음성을 생성하기 위해서는 ssifi.py에서 make_sound함수를 통해 생성하실 수 있습니다.

``make_sound(file_name, sentence, results_path, step=350000)``

- file_name: 생성되는 음성 파일의 이름

- sentence: input text

- result_path: 합성된 음성 파일이 저장되는 경로
- step: 트레이닝 된 모델의 step(default=350000)

상세코드는 ``TTS/synthesize.py`` 를 참고하시면 됩니다.


## Tensorboard
```
tensorboard --logdir log/hp.dataset/
```
- hp.dataset: hparams.py에 등록된 dataset 변수에 등록된 경로
  tensorboard log들은 ```log/hp.dataset/``` directory에 저장됩니다. 그러므로 위의 커멘드를 이용하여 tensorboard를 실행해 학습 상황을 모니터링 하실 수 있습니다.





## References

- NLP
  - [GitHub - SKT-AI/KoGPT2: Korean GPT-2 pretrained cased (KoGPT2)](https://github.com/SKT-AI/KoGPT2)

- TTS
  - [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558), Y. Ren, *et al*.
  - [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263), Y. Ren, *et al*.
  - [ming024's FastSpeech2 impelmentation](https://github.com/ming024/FastSpeech2)
  - [rishikksh20's VocGAN implementation](https://github.com/rishikksh20/VocGAN)
  - [HGU-DLLAB](https://github.com/HGU-DLLAB/Korean-FastSpeech2-Pytorch)
