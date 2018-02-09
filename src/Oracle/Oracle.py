import json

def readPoems():
    word = input("Enter a topic: ")
    word = str.lower(word)
    file = open('poetry.json',encoding = 'UTF-8')
    data = json.load(file)
    for x in data:
        if(str.lower(x['classification'])==word):
            poem = x['text']
            for s in poem:
                print(s)
        for y in x['keywords']:
            if(word==str.lower(y)):
                poem = x['text']
                for s in poem:
                    print(s)
                return
    print("No poem generated")
def main():
    readPoems()


main()
