"""
train_model.py
───────────────
Retraining script for SUCAS's Random Forest complaint classifier.

You already have trained artifacts (rf_model.pkl, tfidf_vectorizer.pkl) —
you don't need to run this to deploy. This script exists so the model is
reproducible: if you get more data later, or need to retrain, this
recreates the exact same pipeline app.py expects.

What this script intentionally does NOT do:
  • It does not train/pickle Logistic Regression, SVM, or Naive Bayes.
    Random Forest is the deployed model, so that's the only one saved.
  • It does not touch the LLM chat assistant (llm_workflow.py). That
    calls an external API at runtime — there's no local model object to
    pickle for it.

Run:
    python train_model.py

Input expected in this folder:
    Final_Dataset.csv   — must contain columns "1" (Department) and "2" (complaint text)

Output written to this folder:
    rf_model.pkl          — trained RandomForestClassifier
    tfidf_vectorizer.pkl  — fitted TfidfVectorizer

Both are saved with joblib (not the plain pickle module) — that's the
format app.py's loader expects, and it's the more robust choice for
sklearn objects that hold numpy arrays internally.
"""

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from text_utils import preprocess

# ──────────────────────────────────────────
# 0. NLTK data (one-time download)
# ──────────────────────────────────────────
for pkg in ("stopwords", "wordnet"):
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def main():
    # ── Load & clean data ──────────────────
    data = pd.read_csv("Final_Dataset.csv")
    df = data[["1", "2"]].rename(columns={"1": "Department", "2": "Complaint"})
    df["Department"] = df["Department"].str.strip()

    print("Cleaning complaint text …")
    df["clean_text"] = df["Complaint"].apply(lambda t: preprocess(t, STOP_WORDS, LEMMATIZER))

    # ── Vectorize ───────────────────────────
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df["clean_text"])
    y = df["Department"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1
    )

    # ── Train Random Forest (the deployed model) ──
    print("Training Random Forest …")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    # ── Save artifacts required by app.py ──
    joblib.dump(model, "rf_model.pkl")
    joblib.dump(tfidf, "tfidf_vectorizer.pkl")

    print("\nSaved: rf_model.pkl, tfidf_vectorizer.pkl")
    print("Logistic Regression / SVM / Naive Bayes were NOT saved (experiment-only, not deployed).")


if __name__ == "__main__":
    main()
