import json
with open('api.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
token=data['token']


import tushare as ts

ts.set_token(token)
pro = ts.pro_api()

df = pro.daily(ts_code='000001.SZ', start_date='20180401', end_date='20241030')

print(df)

import pandas as pd

df.to_csv("daily.csv",index=True)

print(df.head())