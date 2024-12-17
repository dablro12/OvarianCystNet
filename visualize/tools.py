import pandas as pd 

def PID_Check(df):
    pids = df['ID'].apply(lambda x: x.split('_')[1])
    pids = pids.value_counts()
    
    return pids.__len__()

def Labels_Check(df):
    labels_count = df['ID'].apply(lambda x: x.split('_')[0])
    return labels_count.value_counts()


# ID를 Label로 매핑
def map_labels(df):
    """
    ID 컬럼에서 레이블 값을 추출하고,
    이를 숫자로 매핑하여 새로운 칼럼(Label)을 추가합니다.
    0: Benign, 1: Borderline, 2: Malignant
    """
    # 레이블 맵핑 사전 정의
    label_mapping = {
        "0" : "Benign",
        "1" : "Borderline",
        "2" : "Malignant"
    }

    # ID에서 레이블 추출 후 숫자로 매핑
    df['Label'] = df['ID'].apply(lambda x: label_mapping[x.split('_')[0]])
    return df
