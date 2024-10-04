import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load the dataset
df = pd.read_csv('F:/data/spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']

# Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split the dataset
X = df['text']
y = df['label']

# Vectorize the text
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train the model
model = MultinomialNB()
model.fit(X_vectorized, y)

# Save the model and vectorizer
with open('spam_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)
