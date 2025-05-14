# train_model.py

import os
import re
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from tabulate import tabulate

# ========================== CONFIG ==========================
FAST_MODE = True
DATA_PATH = 'data/processed/email_features1.csv'
RESULTS_DIR = 'results'
MODEL_DIR = 'models'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ========================== LOAD DATA ==========================
df = pd.read_csv(DATA_PATH, low_memory=False)
df['label'] = pd.to_numeric(df['label'], errors='coerce')
df.dropna(subset=['label'], inplace=True)
df['subject'] = df['subject'].fillna("")
df['body'] = df['body'].fillna("")
if FAST_MODE:
    df = df.sample(frac=0.3, random_state=42)

# ========================== FEATURE ENGINEERING ==========================
def feature_engineering(df):
    df['subject_len'] = df['subject'].apply(len)
    df['body_len'] = df['body'].apply(len)
    df['num_exclamations'] = df['subject'].str.count('!') + df['body'].str.count('!')
    df['num_uppercase_words'] = df['body'].apply(lambda text: sum(1 for word in text.split() if word.isupper()))
    df['contains_link'] = df['body'].str.contains(r'http[s]?://', case=False, regex=True).astype(int)
    df['contains_html'] = df['body'].str.contains(r'<[^>]+>', case=False, regex=True).astype(int)
    df['reply_to_mismatch'] = (df['reply_to'] != df['from']).astype(int) if 'reply_to' in df.columns and 'from' in df.columns else 0
    df['content_type_html'] = df.get('Content-Type', pd.Series([""] * len(df))).str.contains("html", case=False).astype(int)
    df['sender_domain'] = df['sender'].str.extract(r'@([A-Za-z0-9.-]+)')[0].fillna("").str.lower()
    df['sender_tld'] = df['sender'].str.extract(r'\.([a-z]+)$')[0].fillna("").str.lower()
    df['is_free_email'] = df['sender_domain'].isin(['gmail.com', 'yahoo.com', 'hotmail.com', 'aol.com', 'outlook.com', 'mail.com']).astype(int)
    df['is_foreign_domain'] = df['sender_tld'].isin(['ru', 'cn', 'br', 'xyz']).astype(int)
    return df

df = feature_engineering(df)

numeric_features = [
    'received_count', 'spf_fail', 'hour', 'day_of_week', 'is_weekend', 'is_after_hours', 'urls',
    'subject_len', 'body_len', 'num_exclamations', 'num_uppercase_words', 'contains_link',
    'contains_html', 'reply_to_mismatch', 'content_type_html', 'is_free_email', 'is_foreign_domain'
]
text_features = ['subject', 'body']

X = df[numeric_features + text_features]
y = df['label']

# ========================== DATA SPLIT ==========================
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# ========================== PREPROCESSING ==========================
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
text_transformer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('subj', text_transformer, 'subject'),
    ('body', text_transformer, 'body')
])

def build_pipeline(model):
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

# ========================== GRID SEARCH ==========================
def run_grid_search(name, model, param_grid, cv=3):
    print(f"\n🔍 Grid search for {name}...")
    grid = GridSearchCV(build_pipeline(model), param_grid, scoring='f1', cv=cv, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"✅ Best params for {name}: {grid.best_params_}")
    return grid.best_estimator_

# ========================== PARAM GRIDS ==========================
if FAST_MODE:
    rf_params = {'classifier__n_estimators': [100], 'classifier__max_depth': [10]}
    gb_params = {'classifier__n_estimators': [100], 'classifier__max_depth': [3]}
    mlp_params = {'classifier__hidden_layer_sizes': [(50,)], 'classifier__max_iter': [200]}
    cv_folds = 2
else:
    rf_params = {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [10, 30]}
    gb_params = {'classifier__n_estimators': [100, 150], 'classifier__learning_rate': [0.03, 0.1], 'classifier__max_depth': [3, 5]}
    mlp_params = {'classifier__hidden_layer_sizes': [(50,), (100,)], 'classifier__max_iter': [300], 'classifier__learning_rate_init': [0.001, 0.01]}
    cv_folds = 5

# ========================== TRAIN MODELS ==========================
rf_model = run_grid_search("Random Forest", RandomForestClassifier(class_weight='balanced', random_state=42), rf_params, cv=cv_folds)
gb_model = run_grid_search("Gradient Boosting", GradientBoostingClassifier(random_state=42), gb_params, cv=cv_folds)
mlp_model = run_grid_search("MLP", MLPClassifier(early_stopping=True, random_state=42), mlp_params, cv=cv_folds)

# ========================== STACKING ==========================
stack_model = build_pipeline(
    StackingClassifier(
        estimators=[
            ('rf', rf_model.named_steps['classifier']),
            ('gb', gb_model.named_steps['classifier']),
            ('mlp', mlp_model.named_steps['classifier'])
        ],
        final_estimator=LogisticRegression(max_iter=200),
        passthrough=False,
        cv=cv_folds,
        n_jobs=1
    )
)

models = {
    "Random Forest Tuned": rf_model,
    "Gradient Boosting Tuned": gb_model,
    "MLP Tuned": mlp_model,
    "Stacking Ensemble": stack_model
}

# ========================== EVALUATION ==========================
results = []
plt.figure(figsize=(8, 6))

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append((name, acc, prec, rec, f1))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.4f})")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt_cm = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt_cm.set_title(f'Confusion Matrix: {name}')
    plt_cm.set_xlabel('Predicted')
    plt_cm.set_ylabel('Actual')
    safe_name = re.sub(r'\W+', '_', name.lower())
    plt.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_{safe_name}.png"))
    plt.clf()

# ========================== SAVE ROC CURVE ==========================
plt.figure(figsize=(8, 6))
for name, model in models.items():
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.4f})")

plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "roc_comparison.png"))
plt.close()

# ========================== FEATURE DISTRIBUTIONS ==========================
plt.figure(figsize=(16, 12))
for i, feature in enumerate(numeric_features[:9]):
    plt.subplot(3, 3, i+1)
    sns.boxplot(data=df, x='label', y=feature)
    plt.title(f"Feature: {feature}")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "feature_distributions_part1.png"))
plt.close()

plt.figure(figsize=(16, 12))
for i, feature in enumerate(numeric_features[9:]):
    plt.subplot(3, 3, i+1)
    sns.boxplot(data=df, x='label', y=feature)
    plt.title(f"Feature: {feature}")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "feature_distributions_part2.png"))
plt.close()

# ========================== SAVE MODELS ==========================
df_results = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
best_model = max(results, key=lambda x: x[4])
best_name = best_model[0]
safe_model_name = re.sub(r'\W+', '_', best_name.lower())
best_pipeline = models[best_name]

joblib.dump(best_pipeline, os.path.join(MODEL_DIR, f"best_model_{safe_model_name}.pkl"))
#joblib.dump(best_pipeline, os.path.join(MODEL_DIR, "phishing_model_stacking_ensemble.pkl"))

# Save scaler
scaler = best_pipeline.named_steps['preprocessor'].named_transformers_['num'].named_steps['scaler']
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# Save evaluation summary
print("\n📊 Evaluation Summary:")
print(tabulate(df_results, headers='keys', tablefmt='grid'))
df_results.to_csv(os.path.join(RESULTS_DIR, "evaluation_summary.csv"), index=False)

print(f"\n🏆 Best model: {best_name} | F1 Score: {best_model[4]:.4f}")
print(f"✅ Models and plots saved to: {MODEL_DIR}, {RESULTS_DIR}")
