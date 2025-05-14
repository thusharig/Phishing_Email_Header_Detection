import os
import pandas as pd
import joblib
import glob
import re

# Define model directory
MODEL_DIR = os.path.abspath("C:/Users/labadmin/PycharmProjects/Phishing_Email_Header_Detection/models")

def extract_compatible_features(header):
    subject = header.get("subject", "")
    body = header.get("body", "")
    sender = header.get("sender", "")
    content_type = header.get("Content-Type", "")
    urls = header.get("urls", "0")

    features = {
        'subject': subject,
        'body': body,
        'received_count': 0,
        'spf_fail': 0,
        'hour': 12,
        'day_of_week': 0,
        'is_weekend': 0,
        'is_after_hours': 0,
        'urls': int(urls) if str(urls).isdigit() else 0,
        'subject_len': len(subject),
        'body_len': len(body),
        'num_exclamations': subject.count('!') + body.count('!'),
        'num_uppercase_words': sum(1 for word in body.split() if word.isupper()),
        'contains_link': int('http' in body.lower()),
        'contains_html': int(bool(re.search(r'<[^>]+>', body))),
        'reply_to_mismatch': 0,
        'content_type_html': int("html" in content_type.lower()),
        'is_free_email': int(any(domain in sender.lower() for domain in [
            'gmail.com', 'yahoo.com', 'hotmail.com', 'aol.com', 'outlook.com', 'mail.com'
        ])),
        'is_foreign_domain': int(sender.lower().endswith(('.ru', '.cn', '.br', '.xyz')))
    }

    return pd.DataFrame([features])

# ========================== Main Prediction ==========================
def classify_email_header(header):
    model_files = glob.glob(os.path.join(MODEL_DIR, "best_model_stacking_ensemble.pkl"))
    if not model_files:
        raise FileNotFoundError("❌ No phishing model found in 'models/'")

    model_path = max(model_files, key=os.path.getmtime)
    model = joblib.load(model_path)

    df = extract_compatible_features(header)
    prediction = model.predict(df)[0]
    confidence = model.predict_proba(df)[0][int(prediction)]

    return {
        "label": "Phishing" if prediction == 1 else "Legitimate",
        "confidence": float(round(confidence * 100, 2))
    }
