from flask import Flask, request, jsonify
from flask_cors import CORS
import fasttext
import os

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sentiment_classifier.bin')

if not os.path.exists(MODEL_PATH):
    print(f"Model file not found at {MODEL_PATH}")
    print("Current directory contents:", os.listdir(os.path.dirname(MODEL_PATH)))
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = fasttext.load_model(MODEL_PATH)

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict_sentiment():
    if request.method == 'OPTIONS':
        return '', 204

    data = request.get_json(force=True)
    text = data.get('text', '').strip()
    
    prediction = model.predict(text)
    
    label = prediction[0][0].replace('__label__', '')
    confidence = prediction[1][0]
    
    return jsonify({
        'sentiment': label,
        'confidence': float(confidence)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)