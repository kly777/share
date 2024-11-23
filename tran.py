"""
    将csv文件的连续7行转化为一行，并保存为新的csv文件
"""

import pandas as pd

data = pd.read_csv("daily.csv")

OUTPUT_PATH = "output.csv"

output_df = pd.DataFrame()

for line in range(1, data.shape[0] - 8):
    newline = []
    for i in range(7):
        newline += data.iloc[line + i][3:].tolist()
        newline += [
            data.iloc[line + i].tolist()[6] / data.iloc[line + i + 1].tolist()[6]
        ]

    # 将 newline 作为一行添加到 DataFrame
    newline += [
        data.iloc[line - 1].tolist()[6],
        (data.iloc[line - 1].tolist()[6] / data.iloc[line].tolist()[6] - 1) * 100,
    ]
    new_row = pd.DataFrame([newline])
    output_df = pd.concat([output_df, new_row], ignore_index=True)

# 将 DataFrame 写入 CSV 文件
print(output_df)
output_df.to_csv(OUTPUT_PATH, index=False, header=False)
