import requests
import json

app_id = input("app_id: ")
app_key = input("app_key: ")

language = 'en'

baseurl = 'https://od-api.oxforddictionaries.com:443/api/v1/'
url = baseurl+'/wordlist/'+language+'/lexicalCategory=noun'
lex = 'nouns'
r = requests.get("https://od-api.oxforddictionaries.com:443/api/v1/wordlist/en/lexicalCategory%3Dnoun", headers = {'app_id': app_id, 'app_key': app_key})
print(r.status_code)
jsonfile = open('nouns.json','w')
json.dump(r.json(),jsonfile)
jsonfile.close()
input("enter")
