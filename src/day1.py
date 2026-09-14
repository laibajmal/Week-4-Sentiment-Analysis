import pandas as pd

# Load dataset
df = pd.read_csv("../dataset/sentiment_data.csv")

print("=" * 40)
print("FIRST FIVE RECORDS")
print("=" * 40)
print(df.head())

print("\n" + "=" * 40)
print("DATASET SHAPE")
print("=" * 40)
print(df.shape)

print("\n" + "=" * 40)
print("COLUMN NAMES")
print("=" * 40)
print(df.columns.tolist())

print("\n" + "=" * 40)
print("SENTIMENT COUNT")
print("=" * 40)
print(df["Sentiment"].value_counts())