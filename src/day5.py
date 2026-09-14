import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# Load dataset
df = pd.read_csv("../dataset/sentiment_data.csv")

# Features and labels
X = df["Post"]
y = df["Sentiment"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)

# Train Model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Save trained model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully.")


# Sample Posts
sample_posts = [
    "I really enjoyed this internship.",
    "The class was boring.",
    "Excellent learning experience.",
    "I am unhappy with the service.",
    "The training was average."
]

# Transform posts
sample_vectors = vectorizer.transform(sample_posts)

# Predict sentiment
predictions = model.predict(sample_vectors)

# Predict confidence
probabilities = model.predict_proba(sample_vectors)

print("=" * 75)
print(f"{'POST':40} {'SENTIMENT':15} {'CONFIDENCE'}")
print("=" * 75)

for post, sentiment, prob in zip(sample_posts, predictions, probabilities):
    confidence = max(prob) * 100
    print(f"{post[:40]:40} {sentiment:15} {confidence:.2f}%")
