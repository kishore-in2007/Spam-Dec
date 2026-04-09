import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = os.path.join("data", "malicious_phish", "malicious_phish.csv")
VECTORIZER_OUT = "url_vectorizer.pkl"
MODEL_OUT = "url_model.pkl"
LABELS_OUT = "url_label_encoder.pkl"
MAX_ROWS = int(os.environ.get("MAX_ROWS", "200000"))
MAX_FEATURES = int(os.environ.get("MAX_FEATURES", "120000"))


def main():
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "url" not in df.columns or "type" not in df.columns:
        raise ValueError("Expected columns 'url' and 'type' in dataset.")

    df = df.dropna(subset=["url", "type"])
    df["type"] = df["type"].astype(str).str.lower()

    # Map labels: benign -> safe, everything else -> malicious
    df["label"] = df["type"].apply(lambda t: "benign" if t == "benign" else "malicious")

    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(
        df["url"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=MAX_FEATURES,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    model = LogisticRegression(
        max_iter=1000,
        solver="saga",
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_val_vec)
    print("Accuracy:", accuracy_score(y_val, preds))
    print(classification_report(y_val, preds))

    with open(VECTORIZER_OUT, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model, f)
    # Keep labels file for symmetry even if model uses string labels
    with open(LABELS_OUT, "wb") as f:
        pickle.dump(None, f)

    print("Saved:", VECTORIZER_OUT, MODEL_OUT, LABELS_OUT)


if __name__ == "__main__":
    main()
