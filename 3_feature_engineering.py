import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import pickle

df = pd.read_csv('processed_data.csv')
df.dropna(subset=['cleaned_text'], inplace=True)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = tfidf.fit_transform(df['cleaned_text'])
y = df['label']

selector = SelectKBest(chi2, k=2000)
X_best = selector.fit_transform(X, y)

print(f"Original features: {X.shape[1]}")
print(f"Selected features: {X_best.shape[1]}")

with open('features.pkl', 'wb') as f:
    pickle.dump((X_best, y), f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("\nStage 3 Complete: Features calculated and saved to 'features.pkl'")