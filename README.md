# Week 4 - Sentiment Analysis

This project was completed as part of the internship Week 4 task. It focuses on basic sentiment analysis using Natural Language Processing (NLP) and Machine Learning.

## Work Completed

### Day 1 - Dataset Exploration

* Loaded the sentiment dataset using Pandas.
* Displayed the first five records.
* Checked dataset shape and column names.
* Analyzed sentiment distribution.

### Day 2 - TF-IDF Feature Extraction

* Split the dataset into training and testing sets using an 80/20 split.
* Converted text into numerical features using TF-IDF.
* Used only the training data to fit the TF-IDF vectorizer.
* Transformed the testing data using the same vectorizer.

### Day 3 - Sentiment Prediction

* Trained a Multinomial Naive Bayes classifier.
* Tested the model on sample text.
* Generated Positive, Negative, and Neutral sentiment predictions.

### Day 4 - Model Evaluation

* Evaluated the model using accuracy, classification report, and confusion matrix.
* The test set contained only 2 samples because the dataset contains 10 records.
* The resulting test accuracy was 0.00.
* This result demonstrates the limitation of evaluating a machine learning model on a very small dataset.

### Day 5 - Model Saving and Confidence Prediction

* Saved the trained sentiment model using Joblib.
* Saved the TF-IDF vectorizer for future predictions.
* Generated sentiment predictions with confidence scores.

## Dataset

The dataset contains 10 text posts with three sentiment classes:

* Positive: 4
* Negative: 4
* Neutral: 2

## Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF
* Multinomial Naive Bayes
* Joblib

## Project Structure

Week-4/
|-- dataset/
|   `-- sentiment_data.csv
|-- report/
|-- screenshots/
|-- src/
|   |-- day1.py
|   |-- day2.py
|   |-- day3.py
|   |-- day4.py
|   |-- day5.py
|   |-- sentiment_model.pkl
|   `-- tfidf_vectorizer.pkl
|-- .gitignore
`-- README.md

## How to Run

Open the terminal in the `src` folder and run:

```powershell
python day1.py
python day2.py
python day3.py
python day4.py
python day5.py
```

## Learning Outcomes

This week provided practical experience with:

* Text classification
* TF-IDF vectorization
* Train-test splitting
* Multinomial Naive Bayes
* Model evaluation
* Confusion matrices
* Model persistence using Joblib
* Prediction confidence scores

## Note

The dataset used for this internship exercise is intentionally small. Therefore, the evaluation result should not be considered representative of real-world model performance. A larger and more diverse dataset would be required for reliable evaluation.
