import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder


DATA_PATH = os.environ.get("DATA_PATH", os.path.join("data", "labels_v1.csv"))
VECTORIZER_OUT = os.environ.get("VECTORIZER_OUT", "final_vectorizer.pkl")
MODEL_OUT = os.environ.get("MODEL_OUT", "final_model.pkl")
LABELS_OUT = os.environ.get("LABELS_OUT", "label_encoder.pkl")
MAX_ROWS = int(os.environ.get("MAX_ROWS", "200000"))
MAX_FEATURES = int(os.environ.get("MAX_FEATURES", "80000"))


def normalize_label(val):
    if val is None:
        return None
    v = str(val).strip().lower()
    if v in ["spam", "phish", "phishing"]:
        return "Spam"
    if v in ["not spam", "ham", "legit", "legitimate", "benign", "safe"]:
        return "Not Spam"
    return None


def main():
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Expected columns 'text' and 'label' in dataset.")

    df = df.dropna(subset=["text", "label"])
    df["label_norm"] = df["label"].apply(normalize_label)
    df = df.dropna(subset=["label_norm"])

    if len(df) < 20:
        raise ValueError("Need at least 20 labeled rows to train reliably.")

    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42)

    X = df["text"].astype(str)
    y = df["label_norm"].astype(str)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=MAX_FEATURES,
        strip_accents="unicode",
        lowercase=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_val_enc = encoder.transform(y_val)

    model = LogisticRegression(
        max_iter=1000,
        solver="saga",
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train_vec, y_train_enc)

    preds = model.predict(X_val_vec)
    print("Accuracy:", accuracy_score(y_val_enc, preds))
    print(classification_report(y_val_enc, preds, target_names=encoder.classes_))

    with open(VECTORIZER_OUT, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model, f)
    with open(LABELS_OUT, "wb") as f:
        pickle.dump(encoder, f)

    print("Saved:", VECTORIZER_OUT, MODEL_OUT, LABELS_OUT)


if __name__ == "__main__":
    main()
