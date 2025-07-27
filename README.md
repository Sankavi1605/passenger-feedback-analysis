Passenger Feedback Sentiment & Category Analysis

An NLP system, built with Python and Scikit-learn, to automatically classify passenger feedback by category (e.g., Bus Condition) and sentiment (Positive, Negative, Neutral).

Features
- Preprocesses raw text data for machine learning.
- Classifies feedback into predefined categories.
- Analyzes the sentiment of the feedback.
- Includes an interactive web app built with Streamlit for live predictions.

Technologies Used
- Python
- Scikit-learn
- Pandas
- NLTK
- Streamlit
- Joblib

How to Run
1. Clone the repository: `git clone <your-repo-link>`
2. Install the dependencies: `pip install -r requirements.txt`
3. Run the training script to generate the models: `python train_model.py`
4. Run the Streamlit app: `streamlit run app.py`

*(Note: You will also need a `requirements.txt` file listing your dependencies. You can create it by running `pip freeze > requirements.txt` in your terminal.)*# passenger-feedback-analysis
An NLP system to classify passenger feedback by category and sentiment.
