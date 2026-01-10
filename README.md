# Interview Buddy – KoBERT 기반 인터뷰 AI

Interview Buddy는 AI 허브의 '채용면접 인터뷰 데이터셋(AIHub 데이터셋 SN: 71592)을 활용해 KoBERT 언어 모델을 학습한 인터뷰 평가 시스템입니다.  
실제 채용면접 환경의 질문·답변 텍스트 원천 데이터를 전처리/정제하여 한국어 KoBERT를 인터뷰 답변 분류/평가 특화 모델로 파인튜닝한 프로젝트입니다.

---

## 프로젝트 배경 & 데이터셋

### 사용 데이터셋: 채용면접 인터뷰 데이터 (AIHub SN: 71592)
- 데이터 출처: [AI 허브 채용면접 인터뷰 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71592)
- 데이터 특징:
  - 실제 채용면접과 유사한 질문·답변 음성 → STT 텍스트 변환 + 라벨링  
  - 감정, 의도, 요약 라벨 포함 (언어적 분석 가능)

---

## 주요 기능 (AI 모듈)

- KoBERT 인터뷰 특화 모델 학습 
  AIHub 데이터셋을 정제해 KoBERT 분류기를 학습. 답변 품질, 의도, 감정 등 다중 라벨 예측 가능.

- 원격-로컬 자동 동기화 
  SSH/SCP로 원격 사용자와 로컬 AI 모듈을 연결하는 Bash 오케스트레이터.

- 질문 관리 
  직무/경력별 질문 풀에서 인터뷰 질문을 선택/생성하여 원격 사용자에게 전달.

- 답변 평가 & 피드백
  KoBERT로 답변을 분석하고, 라벨/점수/개선점 등을 포함한 피드백을 생성.

---

## 디렉터리 구조

```text
Interview_Buddy/
├── kobert_train.py         # KoBERT 학습 (AIHub 데이터셋 정제 → 모델 학습)
├── kobert_val.py           # KoBERT 검증/추론
├── kobert_question.py      # 인터뷰 질문 생성/전달
├── kobert_result.py        # 답변 평가 → 피드백 생성
├── kobert_execution.bash   # 원격-로컬 연동 + 전체 파이프라인 자동화
└── data/                   # 데이터셋
```

## 요구사항
- Python / KoBERT:
  - Python 3.8+
  - PyTorch (KoBERT 호환 버전)
  - KoBERT 패키지: transformers, gluonnlp
  - 기타: numpy, pandas, scikit-learn 

- Bash 자동화:
  - Linux + bash, ssh, scp
  - SSH 키 인증 (비밀번호 없이 접속)

---

## KoBERT 모델 학습

AIHub '채용면접 인터뷰 데이터셋을 정제해 KoBERT 분류기를 학습합니다. 

```text
python3 kobert_train.py \
    --train_csv 정제된_학습데이터.csv \
    --val_csv   정제된_검증데이터.csv \
    --model_save_path interview_kobert_model.pth
└── data/                   # 데이터셋
