# ================================
# 📌 1. Ontology 정의 (확장 + None 방지)
# ================================
import re 
import numpy as np

USG_BEHAVIOR_3CLASS = {
    "malignant": [
        "carcinoma", "adenocarcinoma", "cancer", "ovarian ca",
        "ov. ca", "ova ca", "malignant", "malignancy", "malgnancy",
        "clear cell", "sarcoma", "carcinosarcoma", "dysgerminoma",
        "peritoneal carcinoma", "cancerous", "carcinomatous",
        "malginancy", "cacnerous mass", "granulosa cell tumor"
    ],

    "borderline": [
        "borderline", "mucinous borderline",
        "borderline tumor", "borderline malignancy",
        "benign to borderline", "borderline to malignancy",
        "borderline epithelial", "borderline serous",
        "borderline or malignancy"
    ],

    # 🔥 의사가 불확실하게 기재한 KW → 모두 borderline으로 매핑
    "suspicious": [
        "r/o", "rule out", "suspected", "suspect",
        "possible", "likely", "cannot exclude",
        "probable", "undetermined", "uncertain"
    ],

    "benign": [
        "benign", "benign simple",
        "endometrioma", "endometriotic", "endometriosis", "endoemtriosis", "endoteriotic cyst", "endoometrioma", "endomerioma",
        "teratoma", "mature cystic teratoma", "dermoid", "teratom or endometrioma",
        "simple cyst", "benign cyst", "benign mass", "benign tumor",
        "corpus luteal", "mucinous cystadenoma", "serous cystadenoma",
        "fibroma", "myoma", "hydrosalpinx",
        "follicular cyst", "functional cyst",
        "inclusion cyst", "paratubal cyst",
        "bening"
    ]
}
import re

# ------------------------
# Pathology Ontology (3-class)
# ------------------------
PATHOLOGY_3CLASS = {
    "malignant": [
        "high-grade serous carcinoma", "low-grade serous carcinoma",
        "adenocarcinoma", "mucinous carcinoma",
        "endometrioid carcinoma", "carcinosarcoma", "immature teratoma", # 추가
        "germ cell tumor", "yolk sac tumor", "granulosa cell tumor", # 추가 가능
        "malignant", "metastatic", "krukenberg", "lymphoma",
        "poorly differentiated", "strumal carcinoid", "carcinoma",
        "cancer", "leiomyosarcoma", "adenocalcinoma", "clear cell carcinoma"
    ],

    "borderline": [
        "borderline", "seromucinous borderline", "endometrioid borderline",
        "serous borderline", "mucinous borderline",
        "borderline tumor", "borderline malignancy",
        "low malignant potential", "LMP", # 약어 추가
        "with microinvasion", "intraepithelial carcinoma",
        "mitotically active cellular fibroma", "cellular fibroma (borderline)", "borderline"
    ],

    "benign": [
        "mature cystic teratoma", "mature teratoma", 
        "serous cystadenoma", "mucinous cystadenoma",
        "endometrioma", "endometriotic cyst",
        "fibroma", "fibrothecoma", "thecoma", "brenner tumor",
        "corpus luteal cyst", "follicular cyst", "hemorrhagic cyst",
        "paratubal cyst", "inclusion cyst", "simple cyst",
        "tubo-ovarian abscess", "abscess", "endometriosis",
        "nonneoplastic", "non-neoplastic", "benign", "teratoma"
    ]
}

# ------------------------
# Preprocess text
# ------------------------
def preprocess_pathology(t):
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = re.sub(r"[\n\r\t]", " ", t)
    t = re.sub(r"[^\w\s/.-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ------------------------
# Pathology → 3-class mapping
# ------------------------
def map_pathology_to_3class(t):

    # 🔥 0) 가장 먼저 처리해야 하는 규칙
    # borderline 이 포함되면 어떤 경우에도 무조건 borderline(1)
    if "borderline" in t:
        return 1

    # 1) malignant
    for kw in PATHOLOGY_3CLASS["malignant"]:
        if kw in t:
            return 2

    # 2) borderline (추가 규칙)
    for kw in PATHOLOGY_3CLASS["borderline"]:
        if kw in t:
            return 1

    # 3) benign
    for kw in PATHOLOGY_3CLASS["benign"]:
        if kw in t:
            return 0

    # 4) fallback (unknown → benign)
    return 0


def pathology_ontology_parse(text):
    return map_pathology_to_3class(text)


# ================================
# 📌 2. 텍스트 전처리
# ================================
def normalize_txt(df, label_col):
    df = df.copy()
    df.loc[:, label_col] = df[label_col].astype(str).str.lower().str.strip()
    invalid_values = ['', '.', '..', '...', '-', '--']
    df.loc[df[label_col].isin(invalid_values), label_col] = np.nan
    return df


def preprocess_text(t):
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = re.sub(r"[\n\r\t]", " ", t)
    t = re.sub(r"[^\w\s/.-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ================================
# 📌 3. Benign / Borderline / Malignant 매핑
# ================================
def map_usg_to_3class(t):
    t = preprocess_text(t)

    # 4) benign
    for kw in USG_BEHAVIOR_3CLASS["benign"]:
        if kw in t:
            return 0

    # 1) malignant → 최우선 규칙
    for kw in USG_BEHAVIOR_3CLASS["malignant"]:
        if kw in t:
            return 2

    # 2) borderline 명확
    for kw in USG_BEHAVIOR_3CLASS["borderline"]:
        if kw in t:
            return 1

    # 3) 불확실성을 borderline으로
    for kw in USG_BEHAVIOR_3CLASS["suspicious"]:
        if kw in t:
            return 1


    # 5) fallback rules (clinical logic)
    if "cyst" in t:
        return 1
    if "mass" in t and ("solid" not in t):
        return 1
    if "tumor" in t and ("malig" not in t):
        return 1

    # 6) 완전 unknown → borderline
    return 1


# ================================
# 📌 4. 최종 ontology parser
# ================================
def usg_ontology_parse(text):
    t = preprocess_text(text)
    class_id = map_usg_to_3class(t)
    return class_id    # <-- int (0,1,2)를 그대로 반환
