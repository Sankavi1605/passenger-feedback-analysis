# create_dataset.py
import pandas as pd

# create_dataset.py (Updated Data)
data = {
    'feedback_text': [
        "The bus was incredibly clean and the seats were comfortable.",
        "My bus never arrived on time this morning.",
        "The driver was very rude when I asked for a stop.",
        "I appreciate the new air conditioning system, it works great!",
        "The bus smells weird and the windows are dirty.",
        "Neutral feedback regarding the bus schedule, it is as expected.",
        "The bus driver helped an elderly person with their bags. Very kind.",
        "Why is the bus always 15 minutes late? It's unacceptable.",
        "The wifi on the bus is a great feature, but it was not working today.",
        "The price of the ticket is fair for the distance traveled.",
        "The mobile app for tracking buses is fantastic and very accurate.", # New
        "The ticket was a bit expensive for such a short trip."          # New
    ],
    'category': [
        "Bus Condition", "Punctuality", "Driver Behavior", "Bus Condition",
        "Bus Condition", "Punctuality", "Driver Behavior", "Punctuality",
        "Amenities", "Pricing",
        "Amenities",      # New
        "Pricing"         # New
    ],
    'sentiment': [
        "Positive", "Negative", "Negative", "Positive", "Negative",
        "Neutral", "Positive", "Negative", "Neutral", "Positive",
        "Positive",       # New
        "Negative"        # New
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('feedback.csv', index=False)

print("✅ 'feedback.csv' created successfully!")