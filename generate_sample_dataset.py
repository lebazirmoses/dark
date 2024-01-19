import pandas as pd
from sklearn.model_selection import train_test_split

# Sample data
data = {
    'text': [
        "This website tricks users into subscribing to newsletters without their consent.",
        "The checkout process on this website is transparent and user-friendly.",
        "Users are forced to make unintended purchases through misleading buttons.",
        "The website design respects user preferences and promotes clear decision-making.",
        "This app collects personal data without proper disclosure.",
        "Clear and concise privacy policy is provided for users.",
        "Pop-up ads create a sense of urgency, pressuring users to make quick decisions.",
        "The user interface provides helpful information and guides users effectively."
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv("sample_dataset.csv", index=False)

# Split the dataset into train and eval sets
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

# Save train and eval DataFrames to CSV files
train_df.to_csv("train_dataset.csv", index=False)
eval_df.to_csv("eval_dataset.csv", index=False)
