# train_model.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv('dataset_phishing.csv')

# Check the dataset's structure
print(df.head())
print(df.columns)

# Ensure all entries in the URL column are strings
df['url'] = df['url'].astype(str)

# Preprocess data
X = df['url']
y = df['google_index']  # Assuming 'Label' is 1 for phishing and 0 for legitimate

# Feature extraction
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Split data using StratifiedShuffleSplit to ensure both classes in train and test sets
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, test_index in splitter.split(X_vec, y):
    X_train, X_test = X_vec[train_index], X_vec[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Print class distribution in train and test sets
print("Training set class distribution:")
print(y_train.value_counts())
print("Test set class distribution:")
print(y_test.value_counts())

# Train model
model = LogisticRegression(max_iter=1000)  # Increased iterations to ensure convergence
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'model/phishing_model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')
