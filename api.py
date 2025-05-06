from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load('best_rf_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
feature_names = joblib.load('feature_names.pkl')  

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_df = pd.DataFrame([data], columns=feature_names)
    
    prediction = model.predict(input_df)
    outcome = label_encoder.inverse_transform(prediction)[0]
    probabilities = model.predict_proba(input_df)[0]
    outcome_probabilities = dict(zip(label_encoder.classes_, probabilities))
    
    return jsonify({
        'prediction': outcome,
        'probabilities': outcome_probabilities
    })

if __name__ == '__main__':
    app.run(debug=True)