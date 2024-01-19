import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have a dataset with 'dark_pattern' and 'percentage' columns
data = pd.read_csv('dark_pattern_dataset.csv')  # Replace with your dataset

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['dark_pattern'], data['percentage'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train.astype(str))
X_test_tfidf = tfidf_vectorizer.transform(X_test.astype(str))

# Train a Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test_tfidf)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

# Assuming you have a dataset with 'dark_pattern' and 'percentage' columns for visualization
dark_pattern_data = pd.read_csv('dark_pattern_dataset.csv')  # Replace with your dataset

# Plotting the actual vs predicted percentages
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
sns.barplot(x=y_test, y=y_pred)
plt.title('Actual vs Predicted Percentages for Dark Patterns')
plt.xlabel('Actual Percentage')
plt.ylabel('Predicted Percentage')
plt.show()
