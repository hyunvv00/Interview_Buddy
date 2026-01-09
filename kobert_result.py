import os
import json
import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer

ANSWER_DIR = "path to answer"
RESULT_DIR = "path to result"

MODEL_PATH = "runs"
JSON_ROOT = "path to data file/val/labels"

emotion_map = {
    "positive": "긍정",
    "negative": "부정",
    "neutral": "중립"
}
intent_map = {
    "technology": "지식/기술",
    "attitude": "태도",
    "background": "배경",
    "personality": "성향",
    "etc": "기타"
}

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
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "model.pt"), map_location="cpu"))
    model.eval()
    label_encoders = torch.load(os.path.join(MODEL_PATH, "label_encoders.pt"), map_location="cpu")
    tokenizer = AutoTokenizer.from_pretrained('monologg/kobert', trust_remote_code=True)
    return model, tokenizer, label_encoders

def predict_intent_emotion(text, model, tokenizer, label_encoders):
    if not isinstance(text, str):
        text = ""
    encoded = tokenizer(text, padding='max_length', truncation=True, max_length=64, return_tensors='pt')
    with torch.no_grad():
        intent_logits, emotion_logits = model(encoded['input_ids'], encoded['attention_mask'])
    intent = label_encoders['intent'].inverse_transform(torch.argmax(intent_logits, dim=1).cpu().numpy())[0]
    emotion = label_encoders['emotion'].inverse_transform(torch.argmax(emotion_logits, dim=1).cpu().numpy())[0]
    return intent, emotion

def calc_score(pred_intent, pred_emotion, user_text, gt_intent, gt_emotion):
    if not isinstance(user_text, str):
        user_text = ""
    word_count = len(user_text.split())
    if pred_intent == gt_intent and gt_intent:
        intent_score = 30
    elif not gt_intent:
        intent_score = 20
    else:
        similar = [("technology", "attitude"), ("background", "personality")]
        is_similar = any(pred_intent in pair and gt_intent in pair for pair in similar)
        intent_score = 20 if is_similar else 10

    if pred_emotion == gt_emotion and gt_emotion:
        emotion_score = 20
    elif not gt_emotion:
        emotion_score = 15
    else:
        emotion_score = 10

    if 50 <= word_count <= 150:
        length_score = 25
    elif 30 <= word_count < 50:
        length_score = 20
    elif 20 <= word_count < 30:
        length_score = 15
    elif word_count < 20:
        length_score = 5
    else:
        length_score = 10

    quality_score = 15 + (5 if word_count >= 30 else 0) + (5 if pred_intent in ["technology", "attitude"] else 0)
    total = min(100, intent_score + emotion_score + length_score + quality_score)
    return {
        'total': total,
        'intent': intent_score,
        'emotion': emotion_score,
        'length': length_score,
        'quality': quality_score,
        'breakdown': {
            'word_count': word_count,
            'pred_intent': pred_intent,
            'gt_intent': gt_intent,
            'pred_emotion': pred_emotion,
            'gt_emotion': gt_emotion
        }
    }

def get_grade_and_feedback(score_result):
    total = score_result['total']
    word_count = score_result['breakdown']['word_count']
    intent_score = score_result['intent']
    if total >= 90: grade = "A"
    elif total >= 80: grade = "B"
    elif total >= 70: grade = "C"
    elif total >= 60: grade = "D"
    else: grade = "F"
    feedback = []
    if total >= 85: feedback.append("우수한 답변입니다!")
    elif total >= 70: feedback.append("양호한 답변입니다.")
    else: feedback.append("개선이 필요한 답변입니다.")
    if word_count < 20: feedback.append("답변이 너무 짧습니다. 더 구체적으로 작성해보세요.")
    elif word_count > 150: feedback.append("답변이 너무 깁니다. 핵심 위주로 요약하세요.")
    else: feedback.append("적절한 답변 길이입니다.")
    if intent_score >= 25: feedback.append("답변 의도가 잘 표현되었습니다.")
    else: feedback.append("답변의 의도를 더 명확하게 표현해보세요.")
    return grade, feedback

def summarize_text(text):
    try:
        inputs = t5_tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
        summary_ids = t5_model.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)
        return t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        return f"요약 오류: {e}"

def normalize_text(text):
    return text.strip().lower().replace(" ", "").replace(".", "").replace("?", "").replace("!", "")

def find_ground_truth(question_text):
    question_norm = normalize_text(question_text)
    for folder in os.listdir(JSON_ROOT):
        folder_path = os.path.join(JSON_ROOT, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not fname.endswith('.json'):
                continue
            fpath = os.path.join(folder_path, fname)
            try:
                with open(fpath, encoding='utf-8') as f:
                    data = json.load(f)
                qtext = data["dataSet"]["question"]["raw"]["text"]
                qtext_norm = normalize_text(qtext)
                if question_norm in qtext_norm or qtext_norm in question_norm:
                    answer = data["dataSet"]["answer"]
                    gt_emotion = answer.get("emotion", [])
                    gt_intent = answer.get("intent", [])
                    gt_emotion_cat = gt_emotion[0].get("category", "") if gt_emotion else ""
                    gt_intent_cat = gt_intent[0].get("category", "") if gt_intent else ""
                    recommended_answer = answer.get("raw", {}).get("text", "")
                    return {
                        "gt_emotion": gt_emotion_cat,
                        "gt_intent": gt_intent_cat,
                        "recommended_answer": recommended_answer
                    }
            except Exception:
                continue
    return {"gt_emotion": "", "gt_intent": "", "recommended_answer": ""}

def get_user_answer(data):
    if "answer" in data and "user_answer" in data["answer"]:
        user_answer_list = data["answer"]["user_answer"]
        return user_answer_list[0].get("text", "") if user_answer_list else ""
    elif "user_answer" in data:
        user_answer = data["user_answer"]
        if isinstance(user_answer, list):
            return user_answer[0].get("text", "") if user_answer else ""
        elif isinstance(user_answer, dict):
            return user_answer.get("text", "")
        else:
            return user_answer if isinstance(user_answer, str) else ""
    else:
        return ""

def get_question_text(data):
    if isinstance(data.get("question"), dict):
        return data["question"].get("text", "")
    else:
        return data.get("question", "")

def get_user_info(data):
    return data.get("user_info", {})

t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

def analyze_answer_files():
    model, tokenizer, label_encoders = load_model_and_encoders()
    for fname in os.listdir(ANSWER_DIR):
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(ANSWER_DIR, fname)
        with open(fpath, encoding='utf-8') as f:
            data = json.load(f)

        user_answer = get_user_answer(data)
        question_text = get_question_text(data)
        user_info = get_user_info(data)

        gt_info = find_ground_truth(question_text)
        gt_emotion = gt_info["gt_emotion"]
        gt_intent = gt_info["gt_intent"]
        recommended_answer = gt_info["recommended_answer"]

        intent, emotion = predict_intent_emotion(user_answer, model, tokenizer, label_encoders)

        score_result = calc_score(intent, emotion, user_answer, gt_intent, gt_emotion)
        grade, feedback = get_grade_and_feedback(score_result)
        summary = summarize_text(user_answer)

        result_json = {
            "user_info": user_info,
            "question": question_text,
            "user_answer": user_answer,
            "recommended_answer": recommended_answer,
            "score_result": score_result,
            "grade": grade,
            "feedback": feedback,
            "summary": summary,
            "analysis": {
                "pred_emotion_kor": emotion_map.get(emotion, emotion),
                "gt_emotion_kor": emotion_map.get(gt_emotion, gt_emotion),
                "pred_intent_kor": intent_map.get(intent, intent),
                "gt_intent_kor": intent_map.get(gt_intent, intent)
            }
        }
        out_path = os.path.join(RESULT_DIR, f"result_{fname}")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)
        print(f"분석 결과가 {out_path}에 저장되었습니다.")

if __name__ == "__main__":
    analyze_answer_files()
