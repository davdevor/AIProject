from sematch.semantic.similarity import WordNetSimilarity

import codecs

wns = WordNetSimilarity()
poems = codecs.open('generatedpoems.txt','r',encoding='utf-8')
data = open('data.txt','a')
for x in poems:
    temp_words = x.split(" ")
    total = 0
    count = 0
    for y in range(len(temp_words)-1):
        total += wns.word_similarity(temp_words[y], temp_words[y+1], 'li')
        count += 1
    total /= count
    data.write(str(total)+'\n')
data.close()
poems.close()
#print wns.word_similarity(w1, w2, 'li')
