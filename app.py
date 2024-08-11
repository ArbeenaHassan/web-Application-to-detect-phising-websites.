# app.py

from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('model/phishing_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    url = data['url']
    
    # Preprocess input
    url_vec = vectorizer.transform([url])
    
    # Predict
    prediction = model.predict(url_vec)
    result = 'phishing' if prediction[0] == 1 else 'legitimate'
    
    return jsonify({'url': url, 'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
