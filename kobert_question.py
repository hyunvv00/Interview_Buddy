import os
import json
import random

JSON_ROOT = "path to data file/val/labels"
USER_DIR = "path to user"
QUESTION_OUT_DIR = "path to question"

os.makedirs(USER_DIR, exist_ok=True)
os.makedirs(QUESTION_OUT_DIR, exist_ok=True)

OCCUPATION_MAP = {
    'BM': '01.Management',
    'SM': '02.SalesMarketing',
    'PS': '03.PublicService',
    'RND': '04.RND',
    'ICT': '05.ICT',
    'D': '06.Design',
    'PM': '07.ProductionManufacturing'
}

def recommend_question_from_json(user_json_path):
    with open(user_json_path, encoding='utf-8') as f:
        user_data = json.load(f)
    occupation = user_data.get("occupation")
    gender = user_data.get("gender")
    experience = user_data.get("experience")
    channel = user_data.get("channel")
    place = user_data.get("place")
    age_range = user_data.get("ageRange")

    occupation_dir = OCCUPATION_MAP.get(occupation)
    if not occupation_dir:
        print("지원되지 않는 직군입니다.")
        return
    folder_name = f"VL_{occupation_dir}_{gender}_{experience}"
    folder_path = os.path.join(JSON_ROOT, folder_name)
    if not os.path.exists(folder_path):
        print(f"폴더 없음: {folder_path}")
        return

    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    candidates = []
    for fname in files:
        try:
            with open(os.path.join(folder_path, fname), encoding='utf-8') as f:
                data = json.load(f)
            info = data["dataSet"]["info"]
            if (
                (not channel or info.get("channel") == channel)
                and (not place or info.get("place") == place)
                and (not age_range or info.get("ageRange") == age_range)
            ):
                candidates.append((fname, data))
        except Exception as e:
            continue

    if not candidates:
        print("조건에 맞는 질문이 없습니다.")
        return

    _, data = random.choice(candidates)
    question_text = data["dataSet"]["question"]["raw"]["text"]

    question_json = {
        "user_info": user_data,
        "question": {
            "text": question_text
        }
    }

    base_name = os.path.splitext(os.path.basename(user_json_path))[0]
    out_fname = f"q_{base_name}.json"
    out_path = os.path.join(QUESTION_OUT_DIR, out_fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(question_json, f, ensure_ascii=False, indent=2)
    print(f"질문 파일이 {out_path}에 저장되었습니다.")

if __name__ == "__main__":
    for fname in os.listdir(USER_DIR):
        if not fname.endswith('.json'):
            continue
        user_json_path = os.path.join(USER_DIR, fname)
        recommend_question_from_json(user_json_path)
