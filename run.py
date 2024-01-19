import spacy

# Load the spaCy English language model
nlp = spacy.load("en_core_web_sm")

# Define the dark patterns you want to detect
dark_patterns = ["False Urgency", "Basket Sneaking", "Confirm Shaming", "Forced Action", "Subscription Trap", "Interface Interference", "Bait and Switch", "Drip Pricing", "Disguised Advertisement", "Nagging"]

# Function to extract the percentage of each dark pattern
def detect_dark_patterns(text):
    # Create a doc object
    doc = nlp(text)

    # Find named entities, phrases and concepts
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Create a dictionary to store the percentage of each dark pattern
    percentage_dict = {}

    # Count the occurrence of each dark pattern in the entities
    for pattern in dark_patterns:
        percentage_dict[pattern] = round((text.count(pattern)) / len(text.split()) * 100, 2)

    return percentage_dict

# Test the function with some example text
example_text = "In this basket sneaking, we force you to pay before getting your products."
result = detect_dark_patterns(example_text)
print(result)