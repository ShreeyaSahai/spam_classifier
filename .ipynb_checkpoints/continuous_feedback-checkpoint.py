import os
import string
import nltk
import joblib

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def load_ml_object(file_path):
    if not os.path.exists(file_path):
        print(f"Error: Required file '{file_path}' not found. Please run model_training.ipynb first.")
        exit()
    return joblib.load(file_path)

model = load_ml_object('spam_classifier_model.joblib')
vectorizer = load_ml_object('vectorizer.joblib')

def preprocess_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

def predict_message(message):
    clean_message = preprocess_text(message)
    X_input = vectorizer.transform([clean_message])
    prediction = model.predict(X_input)[0]
    print("Prediction:", "Spam" if prediction == 1 else "Ham")
    return prediction, clean_message

def update_model(clean_message, label):
    global model
    X_input = vectorizer.transform([clean_message])
    model.partial_fit(X_input, [label], classes=[0, 1])
    joblib.dump(model, 'spam_classifier_model.joblib')
    print("Model updated and saved.")

def get_valid_label():
    while True:
        try:
            label = int(input("Enter correct label (0 for Ham, 1 for Spam): "))
            if label in [0, 1]:
                return label
            else:
                print("Invalid input. Please enter 0 or 1.")
        except ValueError:
            print("Invalid input. Please enter a number (0 or 1).")

while True:
    print("\n--- Spam Classifier ---")
    print("Options:")
    print("1. Predict a message")
    print("2. Exit")
     
    choice = input("Choose an option (1/2): ").strip()
     
    if choice == '1':
        msg = input("Enter a message: ")
        pred, clean_msg = predict_message(msg)
        feedback = input("Was the prediction correct? (yes/no): ").strip().lower()
        if feedback == 'no':
            correct_label = get_valid_label()
            update_model(clean_msg, correct_label)
        else:
            print("Great! Thank you for your feedback!")
    elif choice == '2':
        print("Exiting...thank you for checking out the spam classifier :) ")
        break
    else:
        print("Invalid choice. Please enter 1 or 2.")