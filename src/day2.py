import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("../dataset/sentiment_data.csv")

# Features and labels
X = df["Post"]
y = df["Sentiment"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("=" * 50)
print("TRAINING DATA")
print("=" * 50)
print(X_train)

print("\n" + "=" * 50)
print("TESTING DATA")
print("=" * 50)
print(X_test)

# Convert text into numerical features
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\n" + "=" * 50)
print("TRAINING DATA SHAPE")
print("=" * 50)
print(X_train_tfidf.shape)

print("\n" + "=" * 50)
print("TESTING DATA SHAPE")
print("=" * 50)
print(X_test_tfidf.shape)

print("\n" + "=" * 50)
print("VOCABULARY SIZE")
print("=" * 50)
print(len(vectorizer.vocabulary_))