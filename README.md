##Sentiment Analysis on Product Reviews##

Overview

This project performs sentiment analysis on product reviews using Naive Bayes Classification. The dataset contains customer reviews with ratings, which are processed to determine whether the sentiment is positive or negative.

Features

Text Preprocessing: Cleaning, tokenization, stopword removal.

Feature Extraction: Bag of Words using CountVectorizer.

Machine Learning Model: Multinomial Naive Bayes.

Evaluation Metrics: Accuracy, Precision, Recall, Confusion Matrix, ROC Curve.

Dataset

The dataset (1429_1.csv) contains:

reviews.rating (numeric ratings from 1-5)

reviews.text (customer review text)

reviews.title (review headline)

reviews.username (customer username)

Installation

Clone the repository and install dependencies:

git clone https://github.com/your-username/sentiment-analysis.git
cd sentiment-analysis
pip install -r requirements.txt

Usage

Run the Jupyter Notebook:

jupyter notebook sentiment_analysis_product.ipynb

Code Breakdown

1. Importing Libraries

import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

2. Preprocessing the Data

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

df['cleaned_text'] = df['reviews.text'].astype(str).apply(clean_text)

3. Converting Text to Features

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])

4. Train-Test Split & Model Training

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

5. Model Evaluation

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

6. Confusion Matrix & ROC Curve

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.show()

Results

Accuracy: ~85%

Precision: 0.87

Recall: 0.82

ROC AUC Score: 0.89

Future Improvements

Use TF-IDF Vectorization for better text representation.

Implement LSTM/BERT for deep learning-based sentiment analysis.

License

This project is licensed under the MIT License.

Feel free to contribute by submitting pull requests or reporting issues!

