# Interview Buddy – KoBERT 기반 인터뷰 AI
<div align="center">

https://github.com/user-attachments/assets/3dc73856-343e-473d-ab89-2d02eadf8fd9

</div>
Interview Buddy는 AI 허브의 '채용면접 인터뷰 데이터셋(AIHub 데이터셋 SN: 71592)을 활용해 KoBERT 언어 모델을 학습한 인터뷰 평가 시스템입니다. 실제 채용면접 환경의 질문·답변 텍스트 원천 데이터를 전처리/정제하여 한국어 KoBERT를 인터뷰 답변 분류/평가 특화 모델로 파인튜닝한 프로젝트입니다.

---

## 핵심 특징
- KoBERT 인터뷰 특화 모델 학습:
  - AIHub 데이터셋을 정제해 KoBERT 분류기를 학습. 답변 품질, 의도, 감정 등 다중 라벨 예측 가능.
- 원격-로컬 자동 동기화:
  - SSH/SCP로 원격 사용자와 로컬 AI 모듈을 연결하는 Bash 오케스트레이터.
- 질문 관리:
  - 직무/경력별 질문 풀에서 인터뷰 질문을 선택/생성하여 원격 사용자에게 전달.
- 답변 평가 & 피드백:
  - KoBERT로 답변을 분석하고, 라벨/점수/개선점 등을 포함한 피드백을 생성.

---

## 데이터셋: 채용면접 인터뷰 데이터 (AIHub SN: 71592)
- 데이터 출처:
  - [AI 허브 채용면접 인터뷰 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71592)
- 데이터 특징:
  - 실제 채용면접과 유사한 질문·답변 음성 → STT 텍스트 변환 + 라벨링  
  - 감정, 의도, 요약 라벨 포함 (언어적 분석 가능)

---

## 디렉터리 구조
```
Interview_Buddy/
├── kobert_train.py         # KoBERT 학습 (AIHub 데이터셋 정제 → 모델 학습)
├── kobert_val.py           # KoBERT 검증/추론
├── kobert_question.py      # 인터뷰 질문 생성/전달
├── kobert_result.py        # 답변 평가 → 피드백 생성
├── kobert_execution.bash   # 원격-로컬 연동 + 전체 파이프라인 자동화
└── data/                   # 데이터셋
```

---

## KoBERT 모델 학습
> AIHub '채용면접 인터뷰 데이터셋을 정제해 KoBERT 분류기를 학습합니다. 
```
python3 kobert_train.py 
    -- train_csv 정제된_학습데이터.csv \
    -- val_csv   정제된_검증데이터.csv \
    -- kobert_model.pth
```

---

## 검증 & 추론
```
python3 kobert_val.py 
    -- kobert_model.pth \
    -- eval_csv 평가데이터.csv \
    -- output_csv 예측결과.csv
```

---

## 파이프라인
```
질문 풀에서 인터뷰 질문을 선택 → question/ 폴더에 저장 → 원격 전송 → answer/ 폴더에서 사용자 답변 로드 → KoBERT 추론 (라벨/점수 산출) → 피드백 생성 → result/ 폴더 저장
```

---

## 질문 생성
> 질문 풀에서 인터뷰 질문을 선택
> question/ 폴더에 저장
> 원격 전송
```
python3 kobert_question.py
```

---

## 답변 평가
> answer/ 폴더에서 사용자 답변 로드
> KoBERT 추론 (라벨/점수 산출)
> 피드백 생성
> result/ 폴더 저장
```
python3 kobert_result.py
```
 
---

## 어노테이션 포맷
> JSON 포맷입니다.

<details>
<summary> ### 사용자 정보 ### </summary>

```json
{
  "occupation": "BM",
  "channel": "MOCK",
  "place": "ONLINE",
  "gender": "Female",
  "ageRange": "-34",
  "experience": "Experienced"
}
```
</details>

<details>
<summary> ### 사용자 질문 ### </summary>

```json
{
  "user_info": {
    "occupation": "BM",
    "channel": "MOCK",
    "place": "ONLINE",
    "gender": "Female",
    "ageRange": "-34",
    "experience": "Experienced"
  },
  "question": {
    "text": "지원한 직무와 관련돼서 특별하게 노력한 게 있다면 말씀해 보시겠습니까."
  }
}
```
</details>

<details>
<summary> ### 사용자 답변 ### </summary>

```json
{
  "user_info": {
    "occupation": "BM",
    "channel": "MOCK",
    "place": "ONLINE",
    "gender": "Female",
    "ageRange": "-34",
    "experience": "Experienced"
  },
  "question": {
    "text": "구성원들과 갈등이 생긴다면 보통 어떻게 해결하실까요"
  },
  "user_answer": 
  [
    {
    "text": "네 답변 드리겠습니다. 저는 구성원들과의 갈등을 최대한 만들지 않으려고 노력하지만 필요할 때는 그 갈등을 좀 더 해결을 해 보려고 노력합니다. 제가 갈등을 해결하는 방식은 처음엔 저의 의견을 논리적으로 설명을 하고 어 상대방이 그것을 좀 더 파악할 수 파악하고 순응할 수 있도록 최선을 다하는 것입니다. 그런데 제 의견이 논리적으로 타당하지 않다를 상대방이 주장한다면 그 의견을 들어보고 제가 맞다고 판단이 되면 그 의견에도 저는 수용을 빨리 하는 편입니다."
    }
  ]
}
```
</details>
 
<details>
<summary> ### 사용자 결과 ### </summary>

```json
{
  "user_info": {
    "occupation": "BM",
    "channel": "MOCK",
    "place": "ONLINE",
    "gender": "Female",
    "ageRange": "-34",
    "experience": "Experienced"
  },
  "question": "구성원들과 갈등이 생긴다면 보통 어떻게 해결하실까요",
  "user_answer": "네 답변 드리겠습니다. 저는 구성원들과의 갈등을 최대한 만들지 않으려고 노력하지만 필요할 때는 그 갈등을 좀 더 해결을 해 보려고 노력합니다. 제가 갈등을 해결하는 방식은 처음엔 저의 의견을 논리적으로 설명을 하고 어 상대방이 그것을 좀 더 파악할 수 파악하고 순응할 수 있도록 최선을 다하는 것입니다. 그런데 제 의견이 논리적으로 타당하지 않다를 상대방이 주장한다면 그 의견을 들어보고 제가 맞다고 판단이 되면 그 의견에도 저는 수용을 빨리 하는 편입니다.",
  "recommended_answer": "네 답변 드리겠습니다. 저는 구성원들과의 갈등을 최대한 만들지 않으려고 노력하지만 필요할 때는 그 갈등을 좀 더 해결을 해 보려고 노력합니다. 제가 갈등을 해결하는 방식은 처음엔 저의 의견을 논리적으로 설명을 하고 어 상대방이 그것을 좀 더 파악할 수 파악하고 순응할 수 있도록 최선을 다하는 것입니다. 그런데 제 의견이 논리적으로 타당하지 않다를 상대방이 주장한다면 그 의견을 들어보고 제가 맞다고 판단이 되면 그 의견에도 저는 수용을 빨리 하는 편입니다.",
  "score_result": {
    "total": 75,
    "intent": 10,
    "emotion": 15,
    "length": 25,
    "quality": 25,
    "breakdown": {
      "word_count": 66,
      "pred_intent": "attitude",
      "gt_intent": "background",
      "pred_emotion": "positive",
      "gt_emotion": ""
    }
  },
  "grade": "C",
  "feedback": [
    "양호한 답변입니다.",
    "적절한 답변 길이입니다.",
    "답변의 의도를 더 명확하게 표현해보세요."
  ],
  "summary": ".              .",
  "analysis": {
    "pred_emotion": "positive",
    "pred_emotion_kor": "긍정",
    "gt_emotion": "",
    "gt_emotion_kor": "",
    "pred_intent": "attitude",
    "pred_intent_kor": "태도",
    "gt_intent": "background",
    "gt_intent_kor": "배경"
  }
}
```
</details>

---
