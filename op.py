    import requests
    from bs4 import BeautifulSoup
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from cryptography.fernet import Fernet
    import pickle

    class DarkPatternDetector:
        def __init__(self):
            self.vectorizer = TfidfVectorizer()
            self.model = RandomForestClassifier()
            self.fernet_key = Fernet.generate_key()
            self.fernet = Fernet(self.fernet_key)

        def train_model(self, X, y):
            try:
                if not X:
                    raise ValueError("Training data (X) is empty.")

                features = self.vectorizer.fit_transform(X)

                if not self.vectorizer.vocabulary_:
                    raise ValueError("Empty vocabulary; perhaps the documents only contain stop words.")

                if features.nnz == 0:
                    raise ValueError("No features were extracted; ensure the documents are not empty or contain only stop words.")

                self.model.fit(features, y)

            except Exception as e:
                print(f"Error during training: {e}")

        def encrypt_model(self):
            try:
                model_state_bytes = pickle.dumps(self.model.__getstate__())
                encrypted_model = self.fernet.encrypt(model_state_bytes)
                return encrypted_model
            except Exception as e:
                print(f"Error during model encryption: {e}")
                return b''  # Return an empty bytes-like object in case of an error

        def decrypt_model(self, encrypted_model):
            try:
                decrypted_model_bytes = self.fernet.decrypt(encrypted_model)
                self.model.__setstate__(pickle.loads(decrypted_model_bytes))
            except Exception as e:
                print(f"Error during model decryption: {e}")

        def extract_features(self, html_content):
            soup = BeautifulSoup(html_content, 'html.parser')
            text_content = soup.get_text(separator=' ')

            if self.vectorizer is not None:
                features = self.vectorizer.transform([text_content])
                return features
            else:
                print("Vectorizer is not initialized.")
                return None

        def detect_dark_pattern(self, url):
            try:
                response = requests.get(url)
                html_content = response.text

                features = self.extract_features(html_content)

                if features is not None:
                    prediction = self.model.predict(features)
                    return bool(prediction[0])
                else:
                    print("Failed to extract features.")
                    return None

            except Exception as e:
                print(f"Error during detection: {e}")
                return None

    # Example usage
    dark_pattern_detector = DarkPatternDetector()
    X_train = ["Your training text 1", "Your training text 2"]
    y_train = [0, 1]  # 0 or 1 depending on whether it's a dark pattern or not
    dark_pattern_detector.train_model(X_train, y_train)

    # Encrypting and saving the model
    encrypted_model = dark_pattern_detector.encrypt_model()
    with open("encrypted_model.dat", "wb") as model_file:
        model_file.write(encrypted_model)

    # Decrypting the model (when needed)
    with open("encrypted_model.dat", "rb") as model_file:
        encrypted_model = model_file.read()
    dark_pattern_detector.decrypt_model(encrypted_model)

    # Example detection
    url_to_check=input("Enter URL : ")
    result = dark_pattern_detector.detect_dark_pattern(url_to_check)

    if result is not None:
        print(f"The website {url_to_check} uses dark patterns: {result}")
    else:
        print("Detection failed.")