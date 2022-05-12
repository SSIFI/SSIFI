import speech_recognition as sr
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt

# 파일 확인
# fig = plt.figure(figsize=(14, 4))
# korean_wav, rate = librosa.core.load('./sound_sample/korean.wav')
# librosa.display.waveshow(korean_wav, sr=rate)
# ipd.Audio(korean_wav, rate=rate)

# 한국어 STT 모델 (GOOGLE)
def speech_recognition(audio_path):
    r = sr.Recognizer()
    audio_file = sr.AudioFile(audio_path)
    with audio_file as source:
        audio = r.record(source)
    return r.recognize_google(audio_data=audio, language='ko-KR')