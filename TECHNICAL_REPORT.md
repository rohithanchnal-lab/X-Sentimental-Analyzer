# Technical Analysis 

This report provides a detailed breakdown of the models used in the X-Sentimental-Analyzer, comparing custom Machine Learning approaches against a lexicon-based baseline.

## 1. Accuracy Benchmarks
* **SVM** achieved the highest overall accuracy at **71%**, providing a significant **8% improvement** over the **VADER** baseline (**63%**).
* While **Naive Bayes** was computationally efficient, its accuracy of **67%** trailed SVM by **4%**.

## 2. Time Comparison
* **The custom ML models were exponentially faster**: **Naive Bayes** completed in just **0.007s** and **SVM** in **0.13s** due to their use of simple probability calculations and vectorized matrix operations.
* **VADER Latency**: In contrast, **VADER** was significantly slower because it must perform a dictionary lookup and intensity calculation for every individual word in the dataset.

## 3. Class-Level Performance
* **SVM Precision**: Demonstrated superior precision for positive sentiment (**0.78**), making it the most reliable model for identifying truly positive posts.
* **Naive Bayes Recall**: Excelled at identifying neutral sentiment with a high recall of **0.84**, though it occasionally misclassified other categories as neutral.
* **VADER Performance**: Showed an impressive recall for positive sentiment (**0.88**) but suffered from low precision (**0.56**), often incorrectly labeling negative or neutral text as positive.

## 4. Technical Conclusion
The project demonstrates that custom-trained ML models using **Chi-Square feature selection** perform better for sentiment tasks, as they significantly reduce noise and achieve higher accuracy. Ultimately, **SVM** provides the best solution for X data by effectively capturing linguistic dependencies that both Naive Bayes and VADER fail to resolve.