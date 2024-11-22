import json
with open('api.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
token=data['token']


