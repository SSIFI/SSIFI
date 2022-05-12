# from STT import STT
# from NLP import Novelbot, Wellnessbot, Basicbot, Painterbot, Reporterbot, Writerbot
from TTS import synthesize
# from NLP import Painterbot

# STT
# audio_path = './STT/audio_sample/korean.wav'
# prompt = STT.speech_recognition(audio_path)
# print(prompt)

# NLP (BASICBOT)
# output = Basicbot.basicbot("강화도의 특산품은 뭐야?", 200)
# print(output)

# NLP (NOVELBOT)
# output = Novelbot.novelbot(prompt, 100)
# print(output)

# NLP (WELLNESSBOT)
# output = Wellnessbot.wellnessbot("외로운 오늘", 50)
# print(output)

# NLP (REPORTERBOT)
# mode 선언 :
# output = Reporterbot.reporterbot("위치추적 전자장치(전자발찌) 훼손 전후로 여성 2명을 잇달아 살해한 ", 200, 'society')
# print(output)

# NLP (WRITERBOT)
# mode 선언 :
# output = Writerbot.writerbot("위치추적 전자장치(전자발찌) 훼손 전후로 여성 2명을 잇달아 살해한 ", 200)
# print(output)

# NLP (PAINTERBOT)
# output = Painterbot.painterbot("곰의 유화")
# print(type(output))

# output = input()
# TTS
results_path = './'
file_name = 'test2'
sentence = input()
step = 540000
synthesize.make_sound(file_name, sentence, results_path, step)
