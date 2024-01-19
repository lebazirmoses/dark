import requests
from bs4 import BeautifulSoup
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from cryptography.fernet import Fernet
import joblib
import os
import pandas as pd

class DarkPatternDetector:
    def __init__(self, centralized_repo_url, encryption_key, vectorizer=None, model_path="dark_pattern_model.joblib", vocab_path="vectorizer_vocab.joblib"):
        self.centralized_repo_url = centralized_repo_url
        self.encryption_key = Fernet(encryption_key)
        self.vectorizer = vectorizer if vectorizer else TfidfVectorizer()
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.load_model()

        # Fit the vectorizer on an example text during initialization
        example_text = "This is an example text to initialize the vectorizer."
        self.vectorizer.fit([example_text])

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)

            # Load the vocabulary only if the file exists
            if os.path.exists(self.vocab_path):
                self.vectorizer.vocabulary_ = joblib.load(self.vocab_path)
            else:
                print("Vectorizer vocabulary file not found. Initializing a new vectorizer.")
        else:
            self.model = RandomForestClassifier(random_state=42)

    def save_model(self):
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer.vocabulary_, self.vocab_path)

    def collect_data(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            html_content = response.text
            return html_content
        except requests.RequestException as e:
            print(f"Error fetching the webpage: {e}")
            return None

    def extract_features(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        text_content = soup.get_text(separator=' ')

        # Use the existing vectorizer or create a new one
        self.vectorizer = self.vectorizer.fit([text_content])

        # Transform the text data using the fitted vectorizer
        features = self.vectorizer.transform([text_content])
        return features

    def encrypt_data(self, data):
        return self.encryption_key.encrypt(data.encode())

    def decrypt_data(self, encrypted_data):
        return self.encryption_key.decrypt(encrypted_data).decode()

    def store_detected_pattern(self, pattern_details):
        # Centralized repository (for illustration purposes)
        encrypted_pattern = self.encrypt_data(pattern_details)
        requests.post(self.centralized_repo_url, data={'pattern': encrypted_pattern})

    def detect_dark_pattern(self, url):
        html_content = self.collect_data(url)

        if html_content is not None:
            # Use the same vectorizer that was used during training
            features = self.extract_features(html_content)

            # Ensure the features have the same shape as during training
            if features.shape[1] == len(self.model.feature_importances_):
                # Mocking a simple dark pattern detection model
                prediction = self.model.predict(features)

                if prediction == 1:
                    print("Potential dark pattern detected on this webpage.")
                    pattern_details = f"Dark Pattern Detected: {url}"
                    self.store_detected_pattern(pattern_details)
                else:
                    print("No dark patterns detected on this webpage.")
            else:
                print("Number of features in the new data does not match the model's expectations.")


def generate_synthetic_dataset():
    # Generate synthetic dataset
    features, labels = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[0.95, 0.05],
        random_state=42
    )

    # Create a DataFrame for better visualization
    columns = [f"Feature_{i}" for i in range(features.shape[1])]
    df = pd.DataFrame(data=features, columns=columns)
    df['Label'] = labels

    # Display a sample of the dataset
    print("Sample of the Synthetic Dataset:")
    print(df.head())

    return features, labels

def train_and_evaluate_model(features, labels, model_path="dark_pattern_model.joblib"):
    # Split the dataset into training and testing sets
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Handle imbalanced datasets using RandomOverSampler
    oversampler = RandomOverSampler(sampling_strategy='minority')
    features_train_resampled, labels_train_resampled = oversampler.fit_resample(features_train, labels_train)

    # Hyperparameter tuning with GridSearchCV
    param_grid = {'n_estimators': [50, 100, 150],
                  'max_depth': [None, 10, 20],
                  'min_samples_split': [2, 5, 10]}
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
    grid_search.fit(features_train_resampled, labels_train_resampled)

    # Retrieve the best model from the grid search
    best_model = grid_search.best_estimator_

    # Make predictions on the test set
    predictions = best_model.predict(features_test)

    # Evaluate the model's accuracy on the test set
    accuracy = accuracy_score(labels_test, predictions)
    print(f"Model Accuracy: {accuracy}")

    # Save the best trained model
    joblib.dump(best_model, model_path)

if __name__ == "__main__":
    # Generate synthetic dataset
    synthetic_features, synthetic_labels = generate_synthetic_dataset()

    # Train and evaluate the model
    train_and_evaluate_model(synthetic_features, synthetic_labels)

    # Use the trained model to detect dark patterns on a new webpage
    # Example Usage
# Example Usage
    centralized_repo_url = "https://example-centralized-repo.com/store-pattern"
    encryption_key = Fernet.generate_key()  # Generates a random Fernet key
    vectorizer = TfidfVectorizer()  # Use the same vectorizer used during training
    detector = DarkPatternDetector(centralized_repo_url, encryption_key, vectorizer)
    url_to_check = "http://example.webscraping.com/"
    detector.detect_dark_pattern(url_to_check)

