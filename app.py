# app.py
import streamlit as st
import joblib

# We can reuse the preprocessing function from our training script
from train_model import preprocess_text

# Use a cache to load the models only once
@st.cache_resource
def load_models():
    """Load and cache the trained models."""
    try:
        category_model = joblib.load('category_model.pkl')
        sentiment_model = joblib.load('sentiment_model.pkl')
        return category_model, sentiment_model
    except FileNotFoundError:
        return None, None

# --- Page Configuration ---
st.set_page_config(
    page_title="Passenger Feedback Analysis",
    page_icon="🚌",
    layout="centered"
)

# --- Load Models ---
category_model, sentiment_model = load_models()

# --- App Interface ---
st.title("Passenger Feedback Analysis 🚌")
st.write(
    "This app uses Machine Learning to analyze passenger feedback. "
    "Enter a comment below to find out its category and sentiment."
)

# Check if models are loaded before showing the text area
if category_model is None or sentiment_model is None:
    st.error(
        "🚨 Model files not found! "
        "Please run `python train_model.py` in your terminal to generate them."
    )
else:
    # Create a form for user input
    with st.form("feedback_form"):
        user_input = st.text_area("Enter passenger feedback here:", height=150)
        submitted = st.form_submit_button("Analyze Feedback")

        if submitted and user_input:
            # 1. Preprocess the input
            processed_input = preprocess_text(user_input)

            # 2. Predict using the loaded models
            predicted_category = category_model.predict([processed_input])[0]
            predicted_sentiment = sentiment_model.predict([processed_input])[0]

            # 3. Display the results
            st.subheader("✨ Analysis Results")
            st.info(f"**Predicted Category:** {predicted_category}")
            st.success(f"**Predicted Sentiment:** {predicted_sentiment}")
        elif submitted and not user_input:
            st.warning("Please enter some feedback before analyzing.")