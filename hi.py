import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Function to analyze webpage content for dark patterns
def analyze_webpage(url):
    # Fetch webpage content
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract text content from the webpage
    text_content = " ".join([p.text for p in soup.find_all('p')])

    # Use TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    text_tfidf = tfidf_vectorizer.transform([text_content])

    # Load pre-trained model (you should train a model using a diverse dataset)
    model = MultinomialNB()
    # Load or train your model here

    # Predict if dark pattern is present
    prediction = model.predict(text_tfidf)

    return prediction[0]

# Example usage
webpage_url = "http://example.webscraping.com/"
result = analyze_webpage(webpage_url)

if result == 1:
    print("Dark pattern detected on the webpage!")
else:
    print("No dark patterns detected.")
