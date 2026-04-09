import os
import csv
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


DATA_SMS_PATH = os.path.join("data", "sms_spam_collection", "SMSSpamCollection")
DATA_LABELS_PATH = os.path.join("data", "labels_v1.csv")
DATA_PHISH_PATH = os.path.join("data", "malicious_phish", "malicious_phish.csv")

VECTORIZER_PATH = "final_vectorizer.pkl"
MODEL_PATH = "final_model.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

URL_VECTORIZER_PATH = "url_vectorizer.pkl"
URL_MODEL_PATH = "url_model.pkl"
URL_LABEL_ENCODER_PATH = "url_label_encoder.pkl"

MAX_ROWS = int(os.environ.get("MAX_ROWS", "100000"))


def load_pickle_any(path):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            import joblib

            return joblib.load(path)


def normalize_label(val):
    if val is None:
        return None
    v = str(val).strip().lower()
    if v in ["spam", "phish", "phishing", "malicious", "malware", "defacement", "scam", "fraud", "suspicious"]:
        return "Spam"
    if v in ["not spam", "ham", "legit", "legitimate", "benign", "safe"]:
        return "Not Spam"
    return None


def read_labels_csv(path):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text")
            label = normalize_label(row.get("label"))
            if text and label:
                texts.append(text)
                labels.append(label)
    return texts, labels


def read_sms_collection(path):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            raw_label, text = parts
            label = normalize_label(raw_label)
            if label and text:
                texts.append(text)
                labels.append(label)
    return texts, labels


def read_malicious_phish(path, max_rows):
    df = pd.read_csv(path)
    if "url" not in df.columns or "type" not in df.columns:
        raise ValueError("Expected columns 'url' and 'type' in malicious_phish.csv")
    df = df.dropna(subset=["url", "type"])
    df["label"] = df["type"].apply(
        lambda t: "Not Spam" if str(t).strip().lower() == "benign" else "Spam"
    )
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)
    return df["url"].tolist(), df["label"].tolist()


def eval_text_model(name, texts, labels, vectorizer, model, label_encoder=None):
    X = vectorizer.transform(texts)
    preds = model.predict(X)
    if label_encoder is not None:
        preds = label_encoder.inverse_transform(preds)
    preds = [normalize_label(p) for p in preds]
    print(f"\n== {name} ==")
    print("Accuracy:", accuracy_score(labels, preds))
    print(classification_report(labels, preds, labels=["Not Spam", "Spam"], zero_division=0))


def eval_url_model(name, urls, labels, url_vectorizer, url_model, url_label_encoder=None):
    X = url_vectorizer.transform(urls)
    preds = url_model.predict(X)
    if url_label_encoder is not None:
        preds = url_label_encoder.inverse_transform(preds)
    preds = [normalize_label(p) for p in preds]
    print(f"\n== {name} ==")
    print("Accuracy:", accuracy_score(labels, preds))
    print(classification_report(labels, preds, labels=["Not Spam", "Spam"], zero_division=0))


def eval_multi_stage(name, texts, labels):
    import app

    preds = []
    for t in texts:
        result, _, _ = app.multi_stage_predict(t)
        preds.append(result)
    print(f"\n== {name} ==")
    print("Accuracy:", accuracy_score(labels, preds))
    print(classification_report(labels, preds, labels=["Not Spam", "Spam"], zero_division=0))


def eval_all_models(name, texts, labels):
    import app

    preds = []
    for t in texts:
        result, _, _ = app.all_models_predict(t)
        preds.append(result)
    print(f"\n== {name} ==")
    print("Accuracy:", accuracy_score(labels, preds))
    print(classification_report(labels, preds, labels=["Not Spam", "Spam"], zero_division=0))


def main():
    # Load models
    vectorizer = load_pickle_any(VECTORIZER_PATH)
    model = load_pickle_any(MODEL_PATH)
    label_encoder = load_pickle_any(LABEL_ENCODER_PATH) if os.path.isfile(LABEL_ENCODER_PATH) else None

    url_vectorizer = load_pickle_any(URL_VECTORIZER_PATH) if os.path.isfile(URL_VECTORIZER_PATH) else None
    url_model = load_pickle_any(URL_MODEL_PATH) if os.path.isfile(URL_MODEL_PATH) else None
    url_label_encoder = load_pickle_any(URL_LABEL_ENCODER_PATH) if os.path.isfile(URL_LABEL_ENCODER_PATH) else None

    # Datasets
    if os.path.isfile(DATA_LABELS_PATH):
        texts, labels = read_labels_csv(DATA_LABELS_PATH)
        if texts:
            eval_text_model("TF-IDF Model on labels_v1.csv", texts, labels, vectorizer, model, label_encoder)
            eval_multi_stage("Multi-stage on labels_v1.csv", texts, labels)
            eval_all_models("All-models (ANY spam) on labels_v1.csv", texts, labels)

    if os.path.isfile(DATA_SMS_PATH):
        texts, labels = read_sms_collection(DATA_SMS_PATH)
        if texts:
            eval_text_model("TF-IDF Model on SMSSpamCollection", texts, labels, vectorizer, model, label_encoder)
            eval_multi_stage("Multi-stage on SMSSpamCollection", texts, labels)
            eval_all_models("All-models (ANY spam) on SMSSpamCollection", texts, labels)

    if url_vectorizer is not None and url_model is not None and os.path.isfile(DATA_PHISH_PATH):
        urls, labels = read_malicious_phish(DATA_PHISH_PATH, MAX_ROWS)
        eval_url_model("URL Model on malicious_phish.csv", urls, labels, url_vectorizer, url_model, url_label_encoder)


if __name__ == "__main__":
    main()
