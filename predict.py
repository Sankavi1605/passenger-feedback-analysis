# predict.py
import joblib
from train_model import preprocess_text # Reuse our preprocessing function

# Load the saved models
try:
    category_model = joblib.load('category_model.pkl')
    sentiment_model = joblib.load('sentiment_model.pkl')
except FileNotFoundError:
    print("Error: Models not found. Please run train_model.py first.")
    exit()

def analyze_feedback(feedback):
    """Analyzes a new piece of feedback using the trained models."""
    processed_feedback = preprocess_text(feedback)
    predicted_category = category_model.predict([processed_feedback])[0]
    predicted_sentiment = sentiment_model.predict([processed_feedback])[0]
    
    print(f"\nOriginal Feedback: '{feedback}'")
    print(f"➡️ Predicted Category: {predicted_category}")
    print(f"➡️ Predicted Sentiment: {predicted_sentiment}")

# --- Test with new examples ---
if __name__ == "__main__":
    print("--- Passenger Feedback Analysis ---")
    analyze_feedback("The driver was speeding and felt unsafe.")
    analyze_feedback("I love the clean buses, thank you!")
    analyze_feedback("The schedule is fine but the app is confusing.")
    # Add your own feedback to test!
    analyze_feedback("The wifi is fast but the seat was broken.")