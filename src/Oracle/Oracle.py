import json

def readPoems():
    word = input("Enter a topic: ")
    file = open('poetry.json',encoding = 'UTF-8')
    data = json.load(file)
    for x in data:
        if(x['classification']==word):
            print(x['text'])
        for y in x['keywords']:
            if(y==word):
                print(x["text"])
                return

def main():
    readPoems()
    input()

main()
