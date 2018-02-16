import requests
import json

app_id = input("app_id: ")
app_key = input("app_key: ")

language = 'en'

baseurl = 'https://od-api.oxforddictionaries.com:443/api/v1/'
url = baseurl+'/wordlist/'+language+'/lexicalCategory=noun'
lex = 'nouns'
r = requests.get("https://od-api.oxforddictionaries.com:443/api/v1/wordlist/en/lexicalCategory%3Dnoun", headers = {'app_id': app_id, 'app_key': app_key})
jsonfile = open('nouns.json','w')
json.dump(r.json(),jsonfile)
jsonfile.close()

r = requests.get("https://od-api.oxforddictionaries.com:443/api/v1/wordlist/en/lexicalCategory%3Dverb", headers = {'app_id': app_id, 'app_key': app_key})
jsonfile = open('verbs.json','w')
json.dump(r.json(),jsonfile)
jsonfile.close()

r = requests.get("https://od-api.oxforddictionaries.com:443/api/v1/wordlist/en/lexicalCategory%3Dadjective", headers = {'app_id': app_id, 'app_key': app_key})
jsonfile = open('adjectives.json','w')
json.dump(r.json(),jsonfile)
jsonfile.close()

r = requests.get("https://od-api.oxforddictionaries.com:443/api/v1/wordlist/en/lexicalCategory%3Dpronoun", headers = {'app_id': app_id, 'app_key': app_key})
jsonfile = open('pronouns.json','w')
json.dump(r.json(),jsonfile)
jsonfile.close()
