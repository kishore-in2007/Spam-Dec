import os
import re
import numpy as np
import pickle
import pytesseract
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from pdf2image import convert_from_path
from werkzeug.utils import secure_filename

# ---------------- PATH SETTINGS ---------------- #

# Optional overrides via environment variables for hosting
TESSERACT_CMD = os.environ.get("TESSERACT_CMD", "").strip()
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Poppler path (needed for PDF OCR on Windows or custom installs)
POPPLER_PATH = os.environ.get("POPPLER_PATH", "").strip() or None

# ---------------- APP INIT ---------------- #

app = Flask(__name__)
CORS(app)

# ---------------- LOAD MODELS ---------------- #

# Legacy TF-IDF + Logistic Regression spam model
VECTORIZER_PATH = "final_vectorizer_v2.pkl"
MODEL_PATH = "final_model_v2.pkl"
LABEL_ENCODER_PATH = "label_encoder_v2.pkl"
URL_VECTORIZER_PATH = "url_vectorizer.pkl"
URL_MODEL_PATH = "url_model.pkl"
URL_LABEL_ENCODER_PATH = "url_label_encoder.pkl"

def load_pickle_any(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return joblib.load(path)


vectorizer = load_pickle_any(VECTORIZER_PATH)
model = load_pickle_any(MODEL_PATH)

label_encoder = None
if os.path.isfile(LABEL_ENCODER_PATH):
    label_encoder = load_pickle_any(LABEL_ENCODER_PATH)

url_vectorizer = None
url_model = None
url_label_encoder = None
if os.path.isfile(URL_VECTORIZER_PATH) and os.path.isfile(URL_MODEL_PATH):
    url_vectorizer = load_pickle_any(URL_VECTORIZER_PATH)
    url_model = load_pickle_any(URL_MODEL_PATH)
    if os.path.isfile(URL_LABEL_ENCODER_PATH):
        url_label_encoder = load_pickle_any(URL_LABEL_ENCODER_PATH)

# ---------------- HOME ---------------- #

@app.route("/")
def home():
    return "SpamDeck Backend is running successfully!"

# ---------------- HELPERS ---------------- #

def predict_text(text):
    if not text or not text.strip():
        return "Not Spam", 0.0

    features = vectorizer.transform([text])
    pred = model.predict(features)[0]

    if not isinstance(pred, str) and label_encoder is not None:
        pred = label_encoder.inverse_transform([pred])[0]

    pred_lower = str(pred).lower().strip()
    if pred_lower in ["spam", "phish", "phishing", "malicious", "scam", "fraud"]:
        result = "Spam"
    elif pred_lower in ["not spam", "ham", "benign", "safe", "legit", "legitimate"]:
        result = "Not Spam"
    else:
        # Fallback to substring checks
        if "not spam" in pred_lower or "ham" == pred_lower:
            result = "Not Spam"
        elif "spam" in pred_lower or "phish" in pred_lower:
            result = "Spam"
        else:
            result = "Not Spam"

    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        try:
            idx = list(model.classes_).index(pred)
        except Exception:
            idx = int(np.argmax(probs))
        confidence = float(probs[idx]) * 100

    return result, round(confidence, 2)


def predict_url_text(text):
    if url_vectorizer is None or url_model is None:
        return None

    if not text or not text.strip():
        return "Not Spam", 0.0

    features = url_vectorizer.transform([text])
    pred = url_model.predict(features)[0]

    if not isinstance(pred, str) and url_label_encoder is not None:
        pred = url_label_encoder.inverse_transform([pred])[0]

    pred_lower = str(pred).lower()
    result = "Spam" if pred_lower not in ["benign", "safe", "legit", "legitimate"] else "Not Spam"

    confidence = 0.0
    if hasattr(url_model, "predict_proba"):
        probs = url_model.predict_proba(features)[0]
        try:
            idx = list(url_model.classes_).index(pred)
        except Exception:
            idx = int(np.argmax(probs))
        confidence = float(probs[idx]) * 100

    return result, round(confidence, 2)


URL_REGEX = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
DOMAIN_REGEX = re.compile(r"\b[a-z0-9.-]+\.[a-z]{2,}(?:/\S*)?\b", re.IGNORECASE)
UPI_ID_REGEX = re.compile(r"\b[a-z0-9._-]{2,}@[a-z]{2,}\b", re.IGNORECASE)
UPI_KEYWORDS = [
    "upi",
    "gpay",
    "google pay",
    "phonepe",
    "paytm",
    "bhim",
    "collect request",
    "request money",
    "refund",
    "cashback",
    "kyc",
    "verify",
    "blocked",
    "suspend",
    "urgent",
    "click",
    "link",
    "account",
    "dear customer",
]


def multi_stage_predict(text):
    base_result, base_conf = predict_text(text)

    text_lower = text.lower() if text else ""
    has_url = bool(URL_REGEX.search(text_lower))
    has_domain = bool(DOMAIN_REGEX.search(text_lower))
    has_upi_id = bool(UPI_ID_REGEX.search(text_lower))

    upi_score = 0
    if has_upi_id:
        upi_score += 2
    if has_url or has_domain:
        upi_score += 1
    for kw in UPI_KEYWORDS:
        if kw in text_lower:
            upi_score += 1

    # Stage 2: URL-specific model if available and input looks like a URL
    if (has_url or has_domain) and url_model is not None and url_vectorizer is not None:
        url_pred = predict_url_text(text)
        if url_pred is not None:
            url_result, url_conf = url_pred
            return url_result, url_conf, {
                "stage": "url_model",
                "signals": {
                    "has_url": has_url,
                    "has_domain": has_domain,
                    "has_upi_id": has_upi_id,
                    "upi_score": upi_score,
                },
            }

    # Heuristic override for likely UPI fraud patterns
    if base_result == "Not Spam" and upi_score >= 3:
        return "Spam", max(base_conf, 60.0), {
            "stage": "upi_heuristic",
            "signals": {
                "has_url": has_url,
                "has_domain": has_domain,
                "has_upi_id": has_upi_id,
                "upi_score": upi_score,
            },
        }

    return base_result, base_conf, {
        "stage": "legacy_model",
        "signals": {
            "has_url": has_url,
            "has_domain": has_domain,
            "has_upi_id": has_upi_id,
            "upi_score": upi_score,
        },
    }


def all_models_predict(text):
    """
    Run all available models and return Spam if any model flags Spam.
    Confidence is the max confidence among the models that ran.
    """
    text_lower = text.lower() if text else ""
    has_url = bool(URL_REGEX.search(text_lower))
    has_domain = bool(DOMAIN_REGEX.search(text_lower))

    results = []

    base_result, base_conf = predict_text(text)
    results.append(("legacy_model", base_result, base_conf))

    if (has_url or has_domain) and url_model is not None and url_vectorizer is not None:
        url_result, url_conf = predict_url_text(text)
        results.append(("url_model", url_result, url_conf))

    # UPI heuristic (keep as a signal model)
    _, _, meta = multi_stage_predict(text)
    if meta.get("stage") == "upi_heuristic":
        results.append(("upi_heuristic", "Spam", max(60.0, base_conf)))

    final_result = "Spam" if any(r[1] == "Spam" for r in results) else "Not Spam"
    final_conf = 0.0
    if results:
        final_conf = max(r[2] for r in results)

    return final_result, round(final_conf, 2), {"stage": "all_models", "results": results}


def extract_text_from_image(image_file):
    img = Image.open(image_file)
    text = pytesseract.image_to_string(img)
    return text


def extract_text_from_pdf(pdf_path):
    pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    full_text = ""
    for page in pages:
        full_text += pytesseract.image_to_string(page) + " "
    return full_text.strip()

# ---------------- ROUTES ---------------- #

@app.route("/predict/spam", methods=["POST"])
def predict_spam():
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")
        result, confidence, meta = multi_stage_predict(text)
        return jsonify({"result": result, "confidence": confidence, **meta})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/url", methods=["POST"])
def predict_url():
    try:
        data = request.get_json(silent=True) or {}
        url = data.get("url", "")
        result, confidence, meta = multi_stage_predict(url)
        return jsonify({"result": result, "confidence": confidence, **meta})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/all", methods=["POST"])
def predict_all():
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")
        result, confidence, meta = all_models_predict(text)
        return jsonify({"result": result, "confidence": confidence, **meta})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/image", methods=["POST"])
def predict_image():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["file"]
        text = extract_text_from_image(file)

        if text.strip() == "":
            return jsonify({"result": "No text found in image", "confidence": 0})

        result, confidence, meta = multi_stage_predict(text)
        return jsonify({"result": result, "confidence": confidence, **meta})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/pdf", methods=["POST"])
def predict_pdf():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No PDF uploaded"}), 400

        file = request.files["file"]
        filename = secure_filename(file.filename)
        temp_path = "temp_" + filename

        file.save(temp_path)

        extracted_text = extract_text_from_pdf(temp_path)
        os.remove(temp_path)

        if extracted_text.strip() == "":
            return jsonify({"result": "No text found in PDF", "confidence": 0})

        result, confidence, meta = multi_stage_predict(extracted_text)
        return jsonify({"result": result, "confidence": confidence, **meta})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- RUN ---------------- #

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(debug=False, host="0.0.0.0", port=port)
