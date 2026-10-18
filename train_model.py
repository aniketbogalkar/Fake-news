"""
NewsCheck AI — Machine Learning Model Training Pipeline
Trains and evaluates Passive-Aggressive Classifier and Logistic Regression on Fake and Real news datasets.
Saves the best model, vectorizer, and evaluation metrics for runtime inference and dashboard metrics.
"""

import os
import re
import json
import string
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def clean_text(text: str) -> str:
    """
    Cleans raw news text and prevents wire-service data leakage:
    - Strips agency header patterns like 'WASHINGTON (Reuters) -' or 'NEW DELHI (PTI) -'
    - Normalizes case, removes URLs, special characters, and extraneous whitespace
    """
    if not isinstance(text, str):
        return ""

    # Remove wire header leakage (common in Reuters, The Hindu, AP, PTI datasets)
    text = re.sub(r'^[A-Z\s,]+ \([A-Za-z\s,]+\)\s*[-—]\s*', '', text)
    text = re.sub(r'^(Reuters|PTI|ANI|The Hindu|BBC)\s*[-—]\s*', '', text, flags=re.IGNORECASE)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Remove HTML tags if present
    text = re.sub(r'<.*?>', ' ', text)

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_dataset():
    """
    Loads True.csv and Fake.csv from the dataset directory.
    Validates schemas and balances classes.
    """
    true_path = os.path.join(DATASET_DIR, "True.csv")
    fake_path = os.path.join(DATASET_DIR, "Fake.csv")

    if not os.path.exists(true_path) or not os.path.exists(fake_path):
        raise FileNotFoundError(
            f"Dataset files missing! Please ensure 'True.csv' and 'Fake.csv' exist in {DATASET_DIR}. "
            f"Run 'python dataset/bootstrap_sample.py' to generate initial sample files."
        )

    print(f"[DataLoader] Reading real news from: {true_path}")
    df_true = pd.read_csv(true_path)
    print(f"[DataLoader] Reading fake news from: {fake_path}")
    df_fake = pd.read_csv(fake_path)

    # Standardize column names
    df_true.columns = [c.lower() for c in df_true.columns]
    df_fake.columns = [c.lower() for c in df_fake.columns]

    # Assign labels: 1 = Real, 0 = Fake
    df_true["label"] = 1
    df_true["label_name"] = "Real"
    df_fake["label"] = 0
    df_fake["label_name"] = "Fake"

    # Merge title and text to enrich context
    for df in [df_true, df_fake]:
        title = df["title"].fillna("") if "title" in df.columns else ""
        text = df["text"].fillna("") if "text" in df.columns else ""
        df["full_content"] = title + " " + text

    # Combine datasets
    df = pd.concat([df_true, df_fake], ignore_index=True)

    # Deduplicate on content to prevent test set data leakage
    initial_len = len(df)
    df = df.drop_duplicates(subset=["full_content"]).reset_index(drop=True)
    print(f"[DataLoader] Loaded {initial_len} articles. After deduplication: {len(df)} articles.")

    # Apply text cleaning
    print("[DataLoader] Cleaning text and removing news agency headers...")
    df["clean_text"] = df["full_content"].apply(clean_text)

    # Filter out empty entries
    df = df[df["clean_text"].str.strip().str.len() > 15].reset_index(drop=True)
    print(f"[DataLoader] Retained {len(df)} valid records ({sum(df['label'] == 1)} Real, {sum(df['label'] == 0)} Fake).")

    return df


def train_and_evaluate():
    """
    Runs TF-IDF feature extraction, trains Passive-Aggressive Classifier and Logistic Regression,
    selects the highest F1-score model, and saves artifacts to models/ directory.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_dataset()

    X = df["clean_text"]
    y = df["label"]

    # Stratified train/test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[Split] Training set: {len(X_train)} samples | Testing set: {len(X_test)} samples")

    # TF-IDF Vectorizer with unigrams and bigrams
    print("[Feature Engineering] Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Candidates to train
    candidates = {
        "Passive-Aggressive Classifier": PassiveAggressiveClassifier(max_iter=1000, random_state=42, C=0.5),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    }

    results = {}
    best_name = None
    best_f1 = -1
    best_model = None

    for name, model in candidates.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"Confusion Matrix: [TN: {cm[0][0]}, FP: {cm[0][1]}, FN: {cm[1][0]}, TP: {cm[1][1]}]")

        results[name] = {
            "accuracy": round(float(acc) * 100, 2),
            "precision": round(float(prec) * 100, 2),
            "recall": round(float(rec) * 100, 2),
            "f1_score": round(float(f1) * 100, 2),
            "confusion_matrix": cm,
            "classification_report": classification_report(y_test, y_pred, target_names=["Fake", "Real"], output_dict=True)
        }

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = model

    print(f"\n[Model Selection] Best performing model: '{best_name}' (F1: {best_f1:.4f})")

    # Save artifacts
    model_path = os.path.join(MODELS_DIR, "model.pkl")
    vectorizer_path = os.path.join(MODELS_DIR, "vectorizer.pkl")
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")

    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    metadata = {
        "best_model_name": best_name,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": results[best_name],
        "all_models_compared": results
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[Artifacts Saved]")
    print(f" - Model: {model_path}")
    print(f" - Vectorizer: {vectorizer_path}")
    print(f" - Evaluation Metrics: {metrics_path}")
    return metadata


if __name__ == "__main__":
    train_and_evaluate()
