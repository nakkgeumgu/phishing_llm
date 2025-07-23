from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
from flask import Flask, request, jsonify
import torch.nn.functional as F
import pandas as pd
import torch
import csv
import os

# ----- 1. 데이터 로딩 및 전처리 -----
df = pd.read_csv("korean_phishing.csv")  # label: 0(정상), 1(피싱)
df["label"] = df["label"].astype(int)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

tokenizer = BertTokenizer.from_pretrained("klue/bert-base")

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

train_ds = train_ds.map(tokenize_function, batched=True)
test_ds = test_ds.map(tokenize_function, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ----- 2. 모델 및 Trainer 초기화 -----
model = BertForSequenceClassification.from_pretrained("klue/bert-base", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)

# 처음 학습 시에만 실행
# trainer.train()

# ----- 3. 예측 함수 -----
def predict(text: str):
    model.eval()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    model.to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        predicted = torch.argmax(logits, dim=1).item()
        return predicted, probs

# ----- 4. Flask API -----
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "text 필드가 필요합니다."}), 400

    label, probs = predict(text)
    return jsonify({
        "label": label,
        "normal_prob": float(probs[0]),
        "phishing_prob": float(probs[1])
    })

# ----- 5. 모델 재학습 (옵션) -----
def retrain_model():
    df = pd.read_csv("korean_phishing.csv")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)

    train_ds = train_ds.map(tokenize_function, batched=True)
    test_ds = test_ds.map(tokenize_function, batched=True)
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    trainer.train_dataset = train_ds
    trainer.eval_dataset = test_ds
    trainer.train()
    print("재학습 완료!")

# ----- 6. 실행 -----
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
