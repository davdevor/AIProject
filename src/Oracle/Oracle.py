import json
import random
def readPoems():
    word = input("Enter a topic: ")
    word = str.lower(word)
    file = open('poetry.json',encoding = 'UTF-8')
    data = json.load(file)
    poems = []
    for x in data:
        if(str.lower(x['classification'])==word):
            poems.append(x['text'])
            continue
        for y in x['keywords']:
            if(word==str.lower(y)):
                poems.append(x['text'])
                continue
    if(len(poems) ==0):
        print("No poem generated")
    else:
        text = poems[random.randrange(0, len(poems)-1)]
        for s in text:
            print(s)
def main():
    readPoems()

main()
