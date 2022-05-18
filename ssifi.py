from STT import STT
from NLP import basicbot, novelbot, wellnessbot, painterbot, kakao_kogpt
from TTS import synthesize

# STT
audio_path = './STT/audio_sample/korean.wav'
prompt = STT.speech_recognition(audio_path)
print(prompt)

# NLP (BASICBOT)
output = Basicbot.basicbot("강화도의 특산품은 뭐야?", 200)
print(output)

# NLP (NOVELBOT)
# output = Novelbot.novelbot(prompt, 100)
# print(output)

# NLP (WELLNESSBOT)
# output = Wellnessbot.wellnessbot("외로운 오늘", 50)
# print(output)

# NLP (PAINTERBOT)
# output = Painterbot.painterbot("곰의 유화")
# print(type(output))

# NLP (KAKAO-kogpt)
# output = kakao_kogpt.kogpt("안녕? 좋은날이야.", 50)
# print(output)

# TTS
results_path = './'
file_name = 'test'
sentence = input()
step = 350000
synthesize.make_sound(file_name, sentence, results_path, step)
