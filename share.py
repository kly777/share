"""
    数据获取
"""

import json
import tushare as ts


with open("api.json", "r", encoding="utf-8") as file:
    data = json.load(file)
token = data["token"]


ts.set_token(token)
pro = ts.pro_api()

df = pro.daily(ts_code="000001.SZ", start_date="20100401", end_date="20241030")

print(df)

df.to_csv("daily.csv", index=True)

print(df.head())
