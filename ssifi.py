from STT import STT
from NLP import basicbot, novelbot, wellnessbot, painterbot, kakao_kogpt
from TTS import synthesize

# STT
audio_path = './STT/audio_sample/korean.wav'
prompt = STT.speech_recognition(audio_path)
print(prompt)

# NLP (BASICBOT)
output = basicbot.basicbot(prompt, 200)
print(output)

# NLP (NOVELBOT)
# output = novelbot.novelbot(prompt, 100)
# print(output)

# NLP (WELLNESSBOT)
# output = wellnessbot.wellnessbot(prompt, 50)
# print(output)

# NLP (PAINTERBOT)
# output = painterbot.painterbot(prompt)
# print(type(output))

# NLP (KAKAO-kogpt)
# output = kakao_kogpt.kogpt(prompt, 50)
# print(output)

# TTS
results_path = './'
file_name = 'test'
sentence = output
step = 350000
synthesize.make_sound(file_name, sentence, results_path, step)
