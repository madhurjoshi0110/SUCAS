"""
text_utils.py
─────────────
Single source of truth for complaint-text cleaning.

Both train_model.py (offline training) and app.py (live inference) import
`preprocess` from here. Keeping one shared function guarantees the model
sees text cleaned the exact same way at train time and at predict time.
"""

import re


def preprocess(text, stop_words, lemmatizer):
    """Lowercase, strip non-letters, remove stopwords, lemmatize, drop tokens <=2 chars."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)
