import json
import random
class SS():
    def __init__(self):
        #the atmoic productions they do not break down  further
        self.atomic = ['Noun','Verb','Pronoun','Adjective','Article','Adverb','Preposition']

        #the dictionary to store the sentence structure
        self.prod = {'Noun':[],
                     'Verb':[],
                     'Adjective':[],
                     'Pronoun':[],
                     'Article':[],
                     'Adverb':[],
                     'Preposition':[]}
        #add to the dictionary the sentence structure
        self.addProd('S','NP VP')
        self.addProd('NP', 'Pronoun|Noun|Article Noun|NP PP')
        self.addProd('VP', 'Verb|VP NP|VP Adjective|VP Adverb')
        self.addProd('PP', 'Preposition NP')
        self.readWords()

    #this method reads the the json files and fills the lists in the prod dictionary
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

        file = open('adverbs.json', mode='r',encoding = 'UTF-8')
        words = json.load(file)
        words= words['results']
        for x in words:
            self.prod['Adverb'].append(x['word'])
        file.close()

        file = open('articles.json', mode='r',encoding = 'UTF-8')
        words = json.load(file)
        words= words['results']
        for x in words:
            self.prod['Article'].append(x['word'])
        file.close()

        file = open('prepositions.json', mode='r',encoding = 'UTF-8')
        words = json.load(file)
        words= words['results']
        for x in words:
            self.prod['Preposition'].append(x['word'])
        file.close()

    def addProd(self,lhs,rhs):
        self.prod[lhs] = rhs.split('|')

    #recursive method to generate sentences
    def generateSentence(self,part):
        #if part given is atomic part then don't recurse further
        if(part in self.atomic):
            #return a random word in the list of the part
            return random.choice(self.prod[part])
        #else get the part and split it into its parts NP VP
        c = random.choice(self.prod[part])
        c = c.split(" ")
        s= ''
        #loop through each part the production
        for x in c:
            #accumulate the return of each recursive call for each part
            s = s + self.generateSentence(x)+ " "
        return s

def main():
    x =  SS()
    for y in range(1,20):
        print(x.generateSentence('S')+'\n')
main()
