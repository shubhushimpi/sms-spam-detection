from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the model and vectorizer
with open('spam_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def spam_prediction():
    text = request.form['text']
    
    # Transform the input text using the same vectorizer
    text_vectorized = vectorizer.transform([text]).toarray()  # Transform and convert to array

    # Make the prediction using the model
    prediction = model.predict(text_vectorized)[0]

    result = "Spam" if prediction == 1 else "Not Spam"
    return render_template('result.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
