import pandas as pd

df = pd.read_csv('train.csv')

print("--- Missing Values Check ---")
print(df.isnull().sum())

df.dropna(subset=['text', 'sentiment'], inplace=True)

label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df['label'] = df['sentiment'].map(label_map)

print("\n--- Data Preview ---")
print(df[['text', 'sentiment', 'label']].head())

df.to_csv('ingested_data.csv', index=False)
print("\nStage 1 Complete: 'ingested_data.csv' created.")