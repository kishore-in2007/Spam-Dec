import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


DATA_PATH = os.path.join("data", "sms_spam_collection", "SMSSpamCollection")
MODEL_NAME = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
OUTPUT_DIR = os.path.join("models", "bert_spam")
MAX_LEN = 128


def read_sms_collection(path):
    texts = []
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            label, text = parts
            label = label.strip().lower()
            if label not in {"ham", "spam"}:
                continue
            texts.append(text.strip())
            labels.append(1 if label == "spam" else 0)
    return texts, labels


def random_typo(text):
    if len(text) < 4:
        return text
    idx = random.randint(1, len(text) - 2)
    ch = text[idx]
    return text[:idx] + ch + ch + text[idx + 1 :]


def obfuscate_spam_words(text):
    mapping = {
        "free": "fr33",
        "win": "w1n",
        "cash": "ca$h",
        "prize": "pr1ze",
        "offer": "0ffer",
        "click": "cl1ck",
        "urgent": "urg3nt",
        "call": "c4ll",
    }
    tokens = text.split()
    out = []
    for t in tokens:
        key = t.lower().strip(".,!?")
        if key in mapping and random.random() < 0.5:
            out.append(mapping[key])
        else:
            out.append(t)
    return " ".join(out)


def add_url_noise(text):
    tails = ["http://bit.ly/4x9Zq", "www.winprize.example", "http://t.co/9ab1"]
    return text + " " + random.choice(tails)


def ocr_noise(text):
    swaps = {"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8"}
    out = []
    for ch in text:
        if random.random() > 0.1:
            out.append(ch)
        else:
            out.append(swaps.get(ch, ch))
    return "".join(out)


def random_delete(text):
    tokens = text.split()
    if len(tokens) <= 3:
        return text
    keep = [t for t in tokens if random.random() > 0.15]
    return " ".join(keep) if keep else text


def augment_text(text, is_spam):
    variants = []
    if is_spam:
        variants.append(obfuscate_spam_words(text))
        variants.append(add_url_noise(text))
    variants.append(ocr_noise(text))
    variants.append(random_typo(text))
    variants.append(random_delete(text))
    return [v for v in variants if v and v != text]


def build_augmented(texts, labels, max_aug_spam=4, max_aug_ham=1):
    out_texts = list(texts)
    out_labels = list(labels)
    for text, label in zip(texts, labels):
        variants = augment_text(text, label == 1)
        random.shuffle(variants)
        max_aug = max_aug_spam if label == 1 else max_aug_ham
        for v in variants[:max_aug]:
            out_texts.append(v)
            out_labels.append(label)
        if label == 1:
            # Extra weight for minority class
            out_texts.append(text)
            out_labels.append(label)
    return out_texts, out_labels


@dataclass
class SpamDataset(torch.utils.data.Dataset):
    encodings: dict
    labels: list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def main():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)

    texts, labels = read_sms_collection(DATA_PATH)
    texts, labels = build_augmented(texts, labels, max_aug_spam=4, max_aug_ham=1)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=7, stratify=labels
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_enc = tokenizer(
        train_texts, truncation=True, padding=True, max_length=MAX_LEN
    )
    val_enc = tokenizer(val_texts, truncation=True, padding=True, max_length=MAX_LEN)

    train_ds = SpamDataset(train_enc, train_labels)
    val_ds = SpamDataset(val_enc, val_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    model.config.label2id = {"ham": 0, "spam": 1}
    model.config.id2label = {0: "ham", 1: "spam"}

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        report_to="none",
    )

    counts = Counter(train_labels)
    total = counts[0] + counts[1]
    class_weights = torch.tensor(
        [total / (2 * counts[0]), total / (2 * counts[1])], dtype=torch.float
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        class_weights=class_weights,
    )

    trainer.train()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved model to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
