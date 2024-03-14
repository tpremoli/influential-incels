import pandas as pd

df = pd.read_csv('sentiment_analysis.csv')

# print the head of the dataframe
print(df.head())

# print the mean of each emotion
print(df.mean())