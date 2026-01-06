from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch
import torch.nn as nn
import random
import os
import json

MODEL_PATH = "runs"
JSON_ROOT = "path to data file/val/labels"

tokenizer = BertTokenizer.from_pretrained('monologg/kobert')

def load_user_data(occupation, gender, experience):
    occupation_mapping = {
        'BM': '01.Management',
        'SM': '02.SalesMarketing',
        'PS': '03.PublicService',
        'RND': '04.RND',
        'ICT': '05.ICT',
        'D': '06.Design',
        'PM': '07.ProductionManufacturing'
    }
    
    if occupation not in occupation_mapping:
        print(f"지원되지 않는 직군: {occupation}")
        return None, None
    
    occupation_dir = occupation_mapping[occupation]
    folder_name = f"VL_{occupation_dir}_{gender}_{experience}"
    folder_path = os.path.join(JSON_ROOT, folder_name)

    if not os.path.exists(folder_path):
        print(f"해당 경로를 찾을 수 없습니다: {folder_path}")
        return None, None
    
    files = os.listdir(folder_path)
    
    if not files:
        print(f"{folder_path} 폴더에 파일이 없습니다.")
        return None, None
    
    random_file = random.choice(files)
    file_path = os.path.join(folder_path, random_file)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    question = data["dataSet"]["question"]["raw"]["text"]
    answer = data["dataSet"]["answer"]["raw"]["text"]
    summary = data["dataSet"]["answer"]["summary"]["text"]
    
    return question, answer, summary

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

def load_model_and_encoders():
    model = MultiOutputKoBERT(intent_classes=5, emotion_classes=3)
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "model.pt")))
    model.eval()
    
    label_encoders = torch.load(os.path.join(MODEL_PATH, "label_encoders.pt"))
    
    return model, tokenizer, label_encoders

def predict_intent_emotion(text, model, tokenizer, label_encoders):
    encoded = tokenizer(text, padding='max_length', truncation=True, max_length=64, return_tensors='pt')
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    
    with torch.no_grad():
        intent_logits, emotion_logits = model(input_ids, attention_mask)
    
    intent = label_encoders['intent'].inverse_transform(torch.argmax(intent_logits, dim=1).cpu().numpy())[0]
    emotion = label_encoders['emotion'].inverse_transform(torch.argmax(emotion_logits, dim=1).cpu().numpy())[0]
    
    return intent, emotion

from transformers import T5ForConditionalGeneration, T5Tokenizer

summarizer = T5ForConditionalGeneration.from_pretrained('t5-small')  
summarizer_tokenizer = T5Tokenizer.from_pretrained('t5-small') 

def summarize_text(text):
    inputs = summarizer_tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    
    summary_ids = summarizer.generate(
        inputs['input_ids'],
        max_length=150,  
        num_beams=4,     
        early_stopping=True
    )
    
    summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

def main():
    print("\n 직군_list")
    print(" BM   : Management")
    print(" SM   : SalesMarketing")
    print(" PS   : PublicService")
    print(" RND  : RND")
    print(" ICT  : ICT")
    print(" D    : Design")
    print(" PM   : ProductionManufacturing")
    occupation = input("\n원하는 직군을 입력하세요 (예: BM, SalesMarketing 등,,,): ")
    gender = input("성별을 입력하세요 (예: Female, Male): ")
    experience = input("경력 여부를 입력하세요 (Experienced, New): ")
    
    question, answer, summary = load_user_data(occupation, gender, experience)
    print(f"\n질문: {question}")
    
    user_answer = input("\n답변을 입력하세요: ")

    model, tokenizer, label_encoders = load_model_and_encoders()

    intent, emotion = predict_intent_emotion(user_answer, model, tokenizer, label_encoders)
    print(f"\n예측된 의도: {intent}")
    print(f"예측된 감정: {emotion}")

    print(f"\n추천 답변: {answer}")
    print(f"\n요약: {summary}")

    summarized_answer = summarize_text(user_answer)
    print(f"\n답변 요약: {summarized_answer}")

if __name__ == "__main__":
    main()
