import pickle
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

with open('features.pkl', 'rb') as f:
    X, y = pickle.load(f)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
target_names = ['Negative', 'Neutral', 'Positive']

def evaluate_and_print(model, name):
    print(f"\n{'='*25} {name} {'='*25}")
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"Train Time: {train_time:.4f}s")
    print(f"Overall Accuracy: {acc:.2%}")
    print("\n--- Precision-Recall Matrix ---")
    print(classification_report(y_test, preds, target_names=target_names))
    
    cm = confusion_matrix(y_test, preds)
    cm_df = pd.DataFrame(cm, index=[f'Actual {n}' for n in target_names], 
                             columns=[f'Pred {n}' for n in target_names])
    print("\n--- Confusion Matrix ---")
    print(cm_df)
    
    return train_time, acc

nb_results = evaluate_and_print(MultinomialNB(), "NAIVE BAYES")
svm_results = evaluate_and_print(LinearSVC(max_iter=1000), "SVM (LINEAR)")

with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(LinearSVC(max_iter=1000).fit(X_train, y_train), f)