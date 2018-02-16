import json
class SS():
    def __init__(self):
        self.prod = {}
        self.addProd('S','NP VP')
        self.addProd('NP', 'Pronoun|Noun')
        self.addProd('VP', 'Verb|VP NP|VP Adjective')
        self.readWords()

    def readWords(self):
        file = open('nouns.json', mode='r',endcoding = 'UTF-8')
        nouns = json.load(file)
        for x in nouns:
            print(x)
    def addProd(self,lhs,rhs):
        self.prod[lhs] = rhs.split('|')

    def generateSentence(self):
        return
