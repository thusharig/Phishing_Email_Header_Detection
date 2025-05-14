import os
from flask import Flask, request, render_template
from flask import Flask, request, render_template, jsonify

import pandas as pd
import joblib
import glob
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# ===== Configure Flask & Template Path =====
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=template_dir)

#  Load Model
MODEL_DIR = r"C:\Users\labadmin\PycharmProjects\Phishing_Email_Header_Detection\models"
model_files = glob.glob(os.path.join(MODEL_DIR, "best_model_stacking_ensemble.pkl"))
if not model_files:
    raise FileNotFoundError("❌ No phishing model found in 'models/'")

model_path = max(model_files, key=os.path.getmtime)
model = joblib.load(model_path)

# Feature Extraction Function
def extract_features(subject, body):
    return pd.DataFrame([{
        'subject': subject,
        'body': body,
        'received_count': 0,
        'spf_fail': 0,
        'urls': body.lower().count('http'),
        'hour': 12,
        'day_of_week': 0,
        'is_weekend': 0,
        'is_after_hours': 0,
        'subject_len': len(subject),
        'body_len': len(body),
        'num_exclamations': subject.count('!') + body.count('!'),
        'num_uppercase_words': sum(1 for w in body.split() if w.isupper()),
        'contains_link': int('http' in body.lower()),
        'contains_html': int(bool(re.search(r'<[^>]+>', body))),
        'reply_to_mismatch': 0,
        'content_type_html': int(bool(re.search(r'<[^>]+>', body))),
        'is_free_email': 0,
        'is_foreign_domain': 0
    }])

# ===== Web Route for Comparison Form and Results =====
@app.route('/', methods=['GET', 'POST'])
def compare():
    legit_result = legit_score = phish_result = phish_score = None

    if request.method == 'POST':
        # Legitimate Email Input
        subject1 = request.form.get('subject1', '')
        body1 = request.form.get('body1', '')

        # Phishing Email Input
        subject2 = request.form.get('subject2', '')
        body2 = request.form.get('body2', '')

        # Legitimate Prediction
        df1 = extract_features(subject1, body1)
        pred1 = model.predict(df1)[0]
        conf1 = model.predict_proba(df1)[0][int(pred1)]
        legit_result = "legitimate" if pred1 == 1 else "Legitimate"
        legit_score = f"Confidence: {round(conf1 * 100, 2)}%"

        # Phishing Prediction
        df2 = extract_features(subject2, body2)
        pred2 = model.predict(df2)[0]
        conf2 = model.predict_proba(df2)[0][int(pred2)]
        phish_result = "Phishing" if pred2 == 1 else "Legitimate"
        phish_score = f"Confidence: {round(conf2 * 100, 2)}%"

    return render_template(
        'index.html',
        legit_result=legit_result,
        legit_score=legit_score,
        phish_result=phish_result,
        phish_score=phish_score
    )

# API route for Chrome extension
@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json()
    subject = data.get("subject", "")
    body = data.get("body", "")

    df = extract_features(subject, body)
    prediction = model.predict(df)[0]
    confidence = model.predict_proba(df)[0][int(prediction)]

    return jsonify({
        "label": "Phishing" if prediction == 1 else "Legitimate",
        "confidence": round(confidence * 100, 2)
    })

# ===== Run the Flask App =====
if __name__ == '__main__':
    app.run(debug=True)
