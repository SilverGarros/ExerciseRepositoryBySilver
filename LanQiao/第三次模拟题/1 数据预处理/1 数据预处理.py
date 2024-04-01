import pandas as pd

# 读取数据集
df = pd.read_csv("./songs_origin.csv")

# 处理缺失值，以其所在列的均值进行填充
df.fillna(df.mean(), inplace=True)
print(df)

# 处理异常值，对于 acousticness_yr 列的值大于 1 或小于 0 的行进行删除
df = df[(df['acousticness_yr'] >= 0) & (df['acousticness_yr'] <= 1)]

print(df)

# 处理重复行，对于数据集中出现多行的相同数据，只保留一行，删除其余重复行
df.drop_duplicates(inplace=True)

# 保存处理后的数据集
df.to_csv('songs_processed.csv', index=False)