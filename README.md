# Simple ML Spam Classifier

A simple machine learning project that classifies messages as **spam** or **ham** (not spam).
This project uses natural language processing (NLP) techniques and handles class imbalance to improve prediction quality.

**Model Accuracy:** \~94%

**Dataset Source:** [SMS Spam Collection Dataset – Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

---

## Features

* Classifies text messages as **Spam** or **Ham**
* Handles **class imbalance** using SMOTE
* Uses **TF-IDF** vectorization for feature extraction
* Built using **Complement Naive Bayes** classifier
* Supports **continuous learning** from user feedback via `partial_fit()`

---

## Dataset Information

The dataset contains 5,574 SMS messages labeled as either:

* `ham` (legitimate message)
* `spam` (unwanted message)
  
---

## Handling Class Imbalance – SMOTE

The dataset has more `ham` messages than `spam`, which can bias the model.

To fix this, I use **SMOTE (Synthetic Minority Over-sampling Technique)**:

* It generates **new, synthetic spam messages** by interpolating between existing ones
* This balances the dataset and improves the model’s ability to detect spam

---

## Preprocessing Steps

Before training, each message goes through the following steps:

1. **Lowercasing** – Converts text to lowercase
2. **Punctuation Removal** – Removes punctuation marks like `!`, `?`, `.`
3. **Tokenization** – Splits sentences into individual words
4. **Lemmatization** – Reduces words to their base form
   *(Uses `WordNetLemmatizer` from NLTK)*
5. **Stopword Removal** – Common words like `"the"`, `"and"`, `"is"` are removed (Handled automatically by `TfidfVectorizer(stop_words='english')`)

---

## Feature Extraction – TF-IDF

`TfidfVectorizer` converts cleaned text into a matrix of numerical values using:

* **TF (Term Frequency):** How often a word appears in a message
* **IDF (Inverse Document Frequency):** How rare a word is across all messages

This helps focus on **important and unique words** in spam detection (e.g., "free", "win", "urgent").

---

## Model – Complement Naive Bayes (ComplementNB)

I use `ComplementNB`, which is well-suited for imbalanced datasets and works efficiently with TF-IDF features.

---

## Continuous Feedback & Learning

The model supports **user feedback** via a command-line interface:

* You enter a message to classify
* If the prediction is wrong, you can correct it
* The model **updates itself immediately** using `partial_fit`, without retraining from scratch

---

## File Overview

* `model_training.ipynb` – Trains the model and saves it using `joblib`
* `continuous_feedback.py` – Loads the model and lets users interact with and update it in real time

---

## How to Run

1. Train the model:
   Run model_training.ipynb

2. Use the feedback loop:

   ```bash
   python continuous_feedback.py
   ```

---

## Requirements

* Python 3.x
* pandas
* scikit-learn
* imbalanced-learn
* nltk
* joblib
