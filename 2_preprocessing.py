import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

df = pd.read_csv('ingested_data.csv')

print("Starting cleaning process...")
df['cleaned_text'] = df['text'].apply(clean_text)

print("\n--- Preprocessing Preview ---")
print(df[['text', 'cleaned_text']].head())

df.to_csv('processed_data.csv', index=False)
print("\nStage 2 Complete: 'processed_data.csv' created.")