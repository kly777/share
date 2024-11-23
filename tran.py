'''
    将csv文件的连续7行转化为一行，并保存为新的csv文件
'''




import pandas as pd

data = pd.read_csv('daily.csv')
# 标签

# 新的 CSV 文件路径
output_file = 'output.csv'
# 创建一个空的 DataFrame
output_df = pd.DataFrame()
for line in range(7, data.shape[0]):
    newline = []
    for i in range(7):
        newline += data.iloc[line - i].tolist()
        newline=newline[1:-3]
    # 将 newline 作为一行添加到 DataFrame
    new_row = pd.DataFrame([newline])
    output_df = pd.concat([output_df, new_row], ignore_index=True)

# 将 DataFrame 写入 CSV 文件
output_df.to_csv(output_file, index=False, header=False)
