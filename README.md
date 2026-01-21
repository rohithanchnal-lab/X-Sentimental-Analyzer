# X Sentiment Analyzer Pipeline

This project was developed as part of the **ESE 590: Practical Machine Learning and Artificial Intelligence** curriculum. A professional machine learning pipeline designed to classify X sentiment as **Positive**, **Negative**, or **Neutral**. This project compares traditional models (SVM & Naive Bayes) against a lexicon-based baseline (VADER). 

## The Pipeline:
1.  **Data Ingestion:** Loading and mapping raw datasets to numerical labels.
2.  **Text Preprocessing:** Advanced cleaning including lemmatization, stop-word removal, and regex-based noise reduction.
3.  **Feature Engineering:** TF-IDF Vectorization and Chi-Square feature selection to identify the most predictive words.
4.  **Model Training:** Comparative training using **Support Vector Machines (SVM)** and **Multinomial Naive Bayes**.
5.  **Benchmarking:** Performance evaluation against the **VADER** lexicon.

## Key Results
- **Model Performance:** SVM consistently outperformed Naive Bayes in handling nuanced social media language.
- **Comparison:** The ML models showed significant accuracy improvements over the rule-based VADER baseline for specific domain slang.

## Tech Stack
- Language: Python
- Libraries: Scikit-learn, NLTK, Pandas, NumPy
- NLP Techniques: TF-IDF, Chi-Square Selection, VADER

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/rohithanchnal-lab/X-Sentimental-Analyzer.git