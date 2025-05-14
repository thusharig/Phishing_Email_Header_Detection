import os
import joblib
import pandas as pd
import glob
import re
import argparse
from datetime import datetime

# ========== Load Latest Model ==========
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
model_files = glob.glob(os.path.join(MODEL_DIR, "best_model_stacking_ensemble.pkl"))
if not model_files:
    raise FileNotFoundError("❌ No model found in 'models' directory.")

latest_model_path = max(model_files, key=os.path.getmtime)
model = joblib.load(latest_model_path)

# ========== Feature Extraction ==========
free_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'aol.com', 'outlook.com', 'mail.com']
foreign_tlds = ['ru', 'cn', 'br', 'xyz']

def extract_features(subject, body, sender, date_str, urls):
    try:
        parsed_date = pd.to_datetime(date_str)
        hour = parsed_date.hour
        day_of_week = parsed_date.dayofweek
    except:
        hour = 12
        day_of_week = 0

    try:
        url_count = int(urls)
    except:
        url_count = 0

    subject_len = len(subject)
    body_len = len(body)
    num_exclamations = subject.count('!') + body.count('!')
    num_uppercase_words = sum(1 for word in body.split() if word.isupper())
    contains_link = int('http' in body.lower())
    contains_html = int(bool(re.search(r'<[^>]+>', body)))
    content_type_html = 1 if "html" in body.lower() else 0

    domain = ""
    tld = ""
    if "@" in sender:
        domain = sender.split("@")[-1].lower()
        tld_match = re.search(r"\.([a-z]+)$", domain)
        tld = tld_match.group(1) if tld_match else ""

    is_free_email = int(domain in free_domains)
    is_foreign_domain = int(tld in foreign_tlds)

    return pd.DataFrame([{
        'subject': subject,
        'body': body,
        'received_count': 0,
        'spf_fail': 0,
        'urls': url_count,
        'hour': hour,
        'day_of_week': day_of_week,
        'is_weekend': int(day_of_week >= 5),
        'is_after_hours': int(hour < 8 or hour > 17),
        'subject_len': subject_len,
        'body_len': body_len,
        'num_exclamations': num_exclamations,
        'num_uppercase_words': num_uppercase_words,
        'contains_link': contains_link,
        'contains_html': contains_html,
        'content_type_html': content_type_html,
        'is_free_email': is_free_email,
        'is_foreign_domain': is_foreign_domain,
        'reply_to_mismatch': 0
    }])

# ========== Argument Parser ==========
parser = argparse.ArgumentParser(description="🔐 Phishing Email Detector CLI")
parser.add_argument('--subject', help='Email subject')
parser.add_argument('--body', help='Email body text')
parser.add_argument('--sender', help='Sender email address')
parser.add_argument('--date', help='Send date (e.g., 2025-05-07 23:15:00)')
parser.add_argument('--urls', help='Number of URLs in the message')
parser.add_argument('--log', action='store_true', help='Log result to CSV')
args = parser.parse_args()

# ========== Interactive Fallback ==========
if not all([args.subject, args.body, args.sender, args.date, args.urls]):
    print("\n📝 Some arguments missing. Switching to interactive mode...\n")
    args.subject = input("Subject: ") if not args.subject else args.subject
    args.body = input("Body: ") if not args.body else args.body
    args.sender = input("Sender Email: ") if not args.sender else args.sender
    args.date = input("Date (e.g., 2025-05-07 23:15:00): ") if not args.date else args.date
    args.urls = input("Number of URLs: ") if not args.urls else args.urls

# ========== Run Prediction ==========
features = extract_features(args.subject, args.body, args.sender, args.date, args.urls)
proba = model.predict_proba(features)[0]
confidence = max(proba)

# Apply manual threshold to reduce false positives
threshold = 0.7
prediction = 1 if proba[1] > threshold else 0
label = "Phishing" if prediction == 1 else "Legitimate"

# ========== Display Output ==========
print("\n🔐 Phishing Detection CLI\n" + "-" * 30)
print(f"📬 Subject: {args.subject}")
print(f"🧾 Prediction: {label}")
print(f"🔎 Confidence: {confidence * 100:.2f}%")

# Optional: Show raw model prediction if you want
# raw_pred = model.predict(features)[0]
# print(f"🤖 Raw model prediction (default threshold): {'Phishing' if raw_pred == 1 else 'Legitimate'}")

# ========== Optional Logging ==========
if args.log:
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "predictions_log.csv")

    log_data = features.copy()
    log_data['Prediction'] = label
    log_data['Confidence'] = round(confidence * 100, 2)
    log_data['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if not os.path.exists(log_file):
        log_data.to_csv(log_file, index=False)
    else:
        log_data.to_csv(log_file, mode='a', header=False, index=False)

    print(f"📝 Prediction logged to: {log_file}")
