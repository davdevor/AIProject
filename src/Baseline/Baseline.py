import json
import random
class SS():
    def __init__(self):
        self.atomic = ['Noun','Verb','Pronoun','Adjective']

        self.prod = {'Noun':[],
                     'Verb':[],
                     'Adjective':[],
                     'Pronoun':[]}
        self.addProd('S','NP VP')
        self.addProd('NP', 'Pronoun|Noun')
        self.addProd('VP', 'Verb|VP NP|VP Adjective')
        self.readWords()

    def readWords(self):
        file = open('nouns.json', mode='r',encoding = 'UTF-8')
        words = json.load(file)
        words= words['results']
        for x in words:
            self.prod['Noun'].append(x['word'])
        file.close()
        
        file = open('verbs.json', mode='r',encoding = 'UTF-8')
        words = json.load(file)
        words= words['results']
        for x in words:
            self.prod['Verb'].append(x['word'])
        file.close()

        file = open('adjectives.json', mode='r',encoding = 'UTF-8')
        words = json.load(file)
        words= words['results']
        for x in words:
            self.prod['Adjective'].append(x['word'])
        file.close()

        file = open('pronouns.json', mode='r',encoding = 'UTF-8')
        words = json.load(file)
        words= words['results']
        for x in words:
            self.prod['Pronoun'].append(x['word'])
        file.close()

    def addProd(self,lhs,rhs):
        self.prod[lhs] = rhs.split('|')

    def generateSentence(self,part):
        if(part in self.atomic):
            return random.choice(self.prod[part])
        c = random.choice(self.prod[part])
        c = c.split(" ")
        s= ''
        for x in c:
            s = s + self.generateSentence(x)+ " "
        return s

def main():
    x =  SS()
    print(x.generateSentence('S'))
main()
