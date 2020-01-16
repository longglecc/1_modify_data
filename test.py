import pandas as pd
import numpy as np

np.random.seed(1)

""" 1. groupby, 按键拆分, 重组, 求和 """
df = pd.DataFrame({
    "key1": ["a", "a", "b", "b", "a"],
    "key2": ["one", "two", "one", "two", "one"],
    "data1": np.random.randint(0, 3, size=5),
    "data2": np.random.randint(0, 3, size=5),
    "data3": np.random.randint(0, 3, size=5),
    "data4": np.random.randint(0, 3, size=5),
    "data5": np.random.randint(0, 3, size=5)
})


df.loc['5'] = 0

print(df)
df_data = df.iloc[:,-5:].copy()
df_balance = df_data
for i in range(1,4,1):

    df1 = df_data.shift(periods=i,axis=1)
    df_balance = pd.concat([df_balance, df1])


df_balance.dropna(axis=1,how='any',inplace=True)
print(df_balance)
print(df_balance.dtypes)
df3 = df_balance.astype("int64")
print(df3.dtypes)



# df1 = df.loc[~(df==0).all(axis=1),:]  #删除
# df2 = df.loc[(df==0).all(axis=1),:]  #找到它
# #df.ix[~(df==0).all(axis=1), :]  # 删了
# print(df1)

# 按key1分组, 计算data1列的平均值
# key1 = df["data1"].groupby(df["key1"]).count()
# print(key1)
#
# # 语法糖为
# key2 = df.groupby(["key1"])["data1"].count()
# print(key2)
#
# key1.reset_index()
# print(key1)
#
#
# print(key1.idxmax())
# print(key1.idxmin())
#
# key3=df[df["key1"]==key1.idxmax()]
# print(key3)
#
# key3.reset_index(drop=True,inplace=True)
# print(key3)
#
# index = key3.index.tolist()
# print(index)



