import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("../dataset/sentiment_data.csv")

# Features and Labels
X = df["Post"]
y = df["Sentiment"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Convert text into numbers
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

print("=" * 50)
print("MODEL TRAINED SUCCESSFULLY")
print("=" * 50)

# Test predictions
sample_posts = [
    "I really enjoyed this movie.",
    "Worst product I have ever used.",
    "The service was okay.",
    "Amazing quality and fast delivery.",
    "Very bad customer support."
]

sample_vectors = vectorizer.transform(sample_posts)
predictions = model.predict(sample_vectors)

print("\nPREDICTIONS")
print("=" * 50)

for post, prediction in zip(sample_posts, predictions):
    print(f"Post: {post}")
    print(f"Predicted Sentiment: {prediction}")
    print("-" * 50)
# Sample posts
sample_posts = [
    "I really enjoyed this internship.",
    "The class was boring.",
    "Excellent learning experience.",
    "I am unhappy with the service."
]

# Convert to TF-IDF
sample_vectors = vectorizer.transform(sample_posts)

# Predict sentiments
predictions = model.predict(sample_vectors)

print("\n" + "=" * 50)
print("PREDICTION RESULTS")
print("=" * 50)

for post, sentiment in zip(sample_posts, predictions):
    print(f"Post      : {post}")
    print(f"Sentiment : {sentiment}")
    print("-" * 50)