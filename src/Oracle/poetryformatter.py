import json
file = open('poetry.json', mode='r', encoding = 'UTF-8')
file2 = open('editedpoems.txt', mode = 'w',encoding = 'UTF-8')
poemjson = json.load(file)
for x in poemjson:
	s = ''
	for y in x['text']:
		s+=y
	file2.write(s)
