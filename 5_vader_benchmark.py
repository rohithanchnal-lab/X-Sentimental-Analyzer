import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('processed_data.csv')
df.dropna(subset=['text', 'label'], inplace=True)

analyzer = SentimentIntensityAnalyzer()

def get_vader_label(text):
    score = analyzer.polarity_scores(str(text))['compound']
    
    if score >= 0.05:
        return 2 
    elif score <= -0.05:
        return 0 
    else:
        return 1 

print("Running VADER benchmark on dataset...")
df['vader_label'] = df['text'].apply(get_vader_label)

vader_acc = accuracy_score(df['label'], df['vader_label'])

print("\n" + "="*25 + " VADER BENCHMARK " + "="*25)
print(f"VADER Overall Accuracy: {vader_acc:.2%}")
print("\n--- VADER Precision-Recall Matrix ---")
print(classification_report(df['label'], df['vader_label'], target_names=['Negative', 'Neutral', 'Positive']))

print("\n--- Sample Comparison (Linguistic Logic) ---")
samples = df[['text', 'sentiment', 'vader_label']].head(5)
print(samples)