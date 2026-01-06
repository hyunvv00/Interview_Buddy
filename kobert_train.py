import os
import json
import pandas as pd  
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from tqdm import tqdm

JSON_ROOT = "path to data file/train"

def extract_input_question_pairs(json_root):
    rows = []
    for root, dirs, files in os.walk(json_root):
        for file in files:
            if file.endswith(".json"):
                try:
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        j = json.load(f)
                    info = j["dataSet"].get("info", {})
                    question = j["dataSet"].get("question", {}).get("raw", {}).get("text", None)
                    if not question:
                        continue
                    occupation = info.get("occupation", "UNKNOWN")
                    gender = info.get("gender", "UNKNOWN")
                    age = info.get("ageRange", "UNKNOWN")
                    experience = info.get("experience", "UNKNOWN")
                    input_text = f"{gender} {occupation} {experience} {age}"
                    rows.append([input_text, question])
                except Exception as e:
                    print(f"⚠️ 오류 in {file}: {e}")
    df = pd.DataFrame(rows, columns=["input_text", "question"])
    df.to_csv("question_generation_dataset.csv", index=False)
    print(f"✅ 질문 생성용 CSV 저장 완료: question_generation_dataset.csv ({len(df)}개 질문)")

def json_to_dataframe(json_root):
    rows = []
    for root, dirs, files in os.walk(json_root):
        for file in files:
            if file.endswith(".json"):
                try:
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        j = json.load(f)
                    answer = j["dataSet"]["answer"]["raw"]["text"]
                    intent = j["dataSet"]["answer"].get("intent", [{}])[0].get("category", "none")
                    emotion = j["dataSet"]["answer"].get("emotion", [{}])[0].get("category", "none")
                    info = j["dataSet"].get("info", {})
                    occupation = info.get("occupation", "UNKNOWN")
                    gender = info.get("gender", "UNKNOWN")
                    age = info.get("ageRange", "UNKNOWN")
                    experience = info.get("experience", "UNKNOWN")
                    rows.append([answer, intent, emotion, occupation, gender, age, experience])
                except Exception as e:
                    print(f"⚠️ 오류 in {file}: {e}")
    df = pd.DataFrame(rows, columns=[
        "text", "label_intent", "label_emotion",
        "occupation", "gender", "ageRange", "experience"
    ])
    df.to_csv("kobert_multi_dataset.csv", index=False)
    print(f"✅ 분석용 KoBERT CSV 저장 완료: kobert_multi_dataset.csv ({len(df)}개 샘플)")
    return df

class MultiOutputKoBERT(nn.Module):
    def __init__(self, intent_classes, emotion_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained("monologg/kobert")
        self.dropout = nn.Dropout(0.3)
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, intent_classes)
        self.emotion_classifier = nn.Linear(self.bert.config.hidden_size, emotion_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        intent_logits = self.intent_classifier(pooled)
        emotion_logits = self.emotion_classifier(pooled)
        return intent_logits, emotion_logits

class MultiTaskDataset(Dataset):
    def __init__(self, texts, intents, emotions, tokenizer, max_len=64):
        self.tokenizer = tokenizer
        self.texts = texts
        self.intents = intents
        self.emotions = emotions
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoded.items()}
        item['intent'] = torch.tensor(self.intents[idx])
        item['emotion'] = torch.tensor(self.emotions[idx])
        return item

def train_multitask_kobert():
    df = json_to_dataframe(JSON_ROOT)

    le_intent = LabelEncoder()
    le_emotion = LabelEncoder()
    df["label_intent_id"] = le_intent.fit_transform(df["label_intent"])
    df["label_emotion_id"] = le_emotion.fit_transform(df["label_emotion"])

    tokenizer = BertTokenizer.from_pretrained("monologg/kobert")

    dataset = MultiTaskDataset(df["text"].tolist(), df["label_intent_id"].tolist(), df["label_emotion_id"].tolist(), tokenizer)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = MultiOutputKoBERT(len(le_intent.classes_), len(le_emotion.classes_))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(3):
        model.train()
        total_loss = 0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent = batch["intent"].to(device)
            emotion = batch["emotion"].to(device)

            optimizer.zero_grad()
            intent_logits, emotion_logits = model(input_ids, attention_mask)
            loss_intent = loss_fn(intent_logits, intent)
            loss_emotion = loss_fn(emotion_logits, emotion)
            loss = loss_intent + loss_emotion
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"\n✅ Epoch {epoch+1} 완료 | 총 Loss: {total_loss:.4f}")

    save_path = "./runs"
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
    tokenizer.save_pretrained(save_path)
    torch.save({"intent": le_intent, "emotion": le_emotion}, os.path.join(save_path, "label_encoders.pt"))
    print(f"\n🎉 학습 완료! 모델이 저장되었습니다: {save_path}")

if __name__ == "__main__":
    extract_input_question_pairs(JSON_ROOT)   
    train_multitask_kobert()     
