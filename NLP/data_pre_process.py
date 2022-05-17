import json, os

# Chat Data
data_path = './data/YOUR_CHAT_DATA.txt'
autoregressive_file = open(data_path, 'w', encoding="UTF-8-sig")
with open(data_path, 'r', encoding="UTF-8-sig") as f:
	data = json.load(f)
	paragraphs = data['data']
	ques_data = ''
	ans_data = ''

	for paragraph in paragraphs:
		paragraph = paragraph['paragraphs'][0]['qas'][0]
		ques_data = paragraph['question']
		ans_data = paragraph['answers'][0]['text']
		autoregressive_file.write(ques_data + '?' + "    " + ans_data + '.' + '\n')

# Autoagressive Data
# 모두의 말뭉치 기반의 Data process
data_path = './data/YOUR_AUTOAGRESSIVE_DATA.txt'
autoregressive_file = open(data_path, 'w', encoding="UTF-8-sig")
for filename in os.listdir(data_path):
	with open(os.path.join(data_path, filename), 'r', encoding="UTF-8-sig") as f:
		datas = json.load(f)
		datas = datas['document'][0]['utterance']
		for data in datas:
			if data['note']:
				autoregressive_file.write("(" + data['note'] + ")" + " ")
			else:
				pass
			if data['form']:
				autoregressive_file.write("'" + data['form'] + "'" + " ")
			else:
				pass
		autoregressive_file.write('\n' + '\n')