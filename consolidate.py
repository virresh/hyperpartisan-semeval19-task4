import pandas as pd

df1 = pd.read_csv('features/prediction.txt')
df2 = pd.read_csv('output/truth.txt')

df = pd.merge(df1, df2, on=['articleId'])
df.to_csv('consolidated.csv', index=None)
