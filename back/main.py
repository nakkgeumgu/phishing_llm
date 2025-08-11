# main.py
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
import os

# ===== 0. 재현성 =====
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ===== 1. 설정 =====
CSV_PATH = "korean_phishing.csv"
CHECKPOINT_DIR = "./results/best"
MODEL_NAME = "klue/bert-base"
LABEL2ID = {"정상": 0, "피싱": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 2. 토크나이저 =====
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# ===== 3. 데이터셋 로드 & 전처리 =====
def load_datasets(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "label"])
    df["labels"] = df["label"].astype(int)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)
    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)

    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

    train_ds = train_ds.map(tokenize_function, batched=True, remove_columns=[c for c in train_ds.column_names if c not in ("labels", "text")])
    test_ds = test_ds.map(tokenize_function, batched=True, remove_columns=[c for c in test_ds.column_names if c not in ("labels", "text")])

    train_ds = train_ds.remove_columns(["text"])
    test_ds = test_ds.remove_columns(["text"])

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return train_ds, test_ds

# ===== 4. 모델 로드 =====
def load_model():
    if os.path.isdir(CHECKPOINT_DIR) and os.path.exists(os.path.join(CHECKPOINT_DIR, "config.json")):
        print("[INFO] 저장된 모델 로드")
        model = BertForSequenceClassification.from_pretrained(
            CHECKPOINT_DIR, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
        )
    else:
        print("[INFO] 저장된 모델 없음 → 새 모델 생성")
        model = BertForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
        )
    return model

model = load_model().to(device)

# ===== 5. Trainer 생성 =====
def make_trainer(train_ds, test_ds):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        acc = accuracy_score(p.label_ids, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            p.label_ids, preds, average="binary", pos_label=1, zero_division=0
        )
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    return trainer

# ===== 6. 최초 실행 시 학습 =====
if not (os.path.isdir(CHECKPOINT_DIR) and os.path.exists(os.path.join(CHECKPOINT_DIR, "config.json"))):
    print("[INFO] 최초 실행 - 학습 시작")
    train_ds, test_ds = load_datasets(CSV_PATH)
    trainer = make_trainer(train_ds, test_ds)
    trainer.train()
    trainer.save_model(CHECKPOINT_DIR)
    tokenizer.save_pretrained(CHECKPOINT_DIR)
    print("[INFO] 모델 저장 완료")

    # 학습된 모델 다시 로드
    model = load_model().to(device)

# ===== 7. 추론 함수 =====
@torch.inference_mode()
def predict(text: str):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    logits = model(**inputs).logits
    probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred = int(torch.argmax(logits, dim=1).item())
    return pred, probs

# ===== 8. Flask API =====
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "")
        date = data.get("date", "")

        if not text or not date:
            return jsonify({"error": "text와 date 필드는 필수입니다."}), 400

        label, probs = predict(text)
        return jsonify({
            "date": date,
            "content": text,
            "label": label,
            "phishing_prob": float(probs[1])
        })
    except Exception:
        return jsonify({"error": "internal error"}), 500

# ===== 9. 실행 =====
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
