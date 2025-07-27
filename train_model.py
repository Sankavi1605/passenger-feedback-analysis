# train_model.py (Updated Version)

import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def preprocess_text(text):
    """Cleans and prepares text data for modeling."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    # Updated this line to fix a deprecation warning
    text = re.sub(r'[^a-zA-Z\s]', '', text, flags=re.I|re.A)
    text = text.lower()
    tokens = word_tokenize(text)
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(processed_tokens)

if __name__ == "__main__":
    # --- 1. Load and Preprocess Data ---
    df = pd.read_csv('feedback.csv')
    df['processed_text'] = df['feedback_text'].apply(preprocess_text)
    print("✅ Data preprocessing complete.")

    # --- 2. Define Features (X) and Targets (y) ---
    X = df['processed_text']
    y_category = df['category']
    y_sentiment = df['sentiment']

    # --- 3. Build and Train Category Model ---
    print("\n⏳ Training Category Classification Model...")
    # Updated test_size from 0.2 to 0.5
    X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
        X, y_category, test_size=0.5, random_state=42, stratify=y_category
    )
    pipeline_category = Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(1, 2))),
        ('classifier', MultinomialNB())
    ])
    pipeline_category.fit(X_train_cat, y_train_cat)
    print("✅ Category model trained.")

    # --- 4. Build and Train Sentiment Model ---
    print("\n⏳ Training Sentiment Analysis Model...")
    # Updated test_size from 0.2 to 0.5 for consistency
    X_train_sent, X_test_sent, y_train_sent, y_test_sent = train_test_split(
        X, y_sentiment, test_size=0.5, random_state=42, stratify=y_sentiment
    )
    pipeline_sentiment = Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(1, 2))),
        ('classifier', MultinomialNB())
    ])
    pipeline_sentiment.fit(X_train_sent, y_train_sent)
    print("✅ Sentiment model trained.")

    # --- 5. Evaluate and Save Models ---
    print("\n--- Category Model Evaluation ---")
    y_pred_cat = pipeline_category.predict(X_test_cat)
    print(classification_report(y_test_cat, y_pred_cat, zero_division=0))

    print("\n--- Sentiment Model Evaluation ---")
    y_pred_sent = pipeline_sentiment.predict(X_test_sent)
    print(classification_report(y_test_sent, y_pred_sent, zero_division=0))

    joblib.dump(pipeline_category, 'category_model.pkl')
    joblib.dump(pipeline_sentiment, 'sentiment_model.pkl')
    print("\n✅ Models saved as 'category_model.pkl' and 'sentiment_model.pkl'")