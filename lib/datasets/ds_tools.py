import numpy as np 
import os 
import pandas as pd 
from lib.datasets.sampler import multilabel_stratified_kfold as ms_kfold
from lib.datasets.validation import validate_pid_distributions, print_label_distribution

def _cnt_data(DATA_DIR):
    """ 1. Data 수 확인 """
    data_paths = []
    for dir, _, files in os.walk(DATA_DIR):
        for file in files:
            data_path = os.path.join(dir, file)
            data_paths.append(data_path)
    print(f"디렉토리 내 파일 개수 : {len(data_paths)}개")
    
    return data_paths

def _filter_data(li1, li2):
    """ 
    2. Filtering *JSON파일이 더 많음 
    JSON 파일이 더 많아서 이미지와 JSON이 같은 이름을 가진 파일만 추출하는 함수 
    """
    li1 = [os.path.basename(file) for file in li1]
    li2 = [os.path.basename(file) for file in li2]
    
    li1 = set(li1)
    li2 = set(li2)
    
    #png 제거하기 
    li1 = [file.replace(".png", "") for file in li1]
    li2 = [file.replace(".json", "") for file in li2]
    
    filtered_li = list(np.intersect1d(li1, li2))
    print(f"필터링 후 파일 개수 : {len(filtered_li)}개 / {len(li2)}개")
    return filtered_li



def _label_patient_split(fi_datanames): 
    """ 
    3. 0,1,2 구분
    EX : 0_R001_00001 -> {Label}_{PID}_{Img Number}
    """
    pid_dict = {"0" : {}, "1" : {}, "2" : {}}
    for pid in fi_datanames:
        label = pid.split('_')[0]
        PID = pid.split('_')[1]
        if PID in pid_dict[label]:
            pid_dict[label][PID].append(pid)
        else:
            pid_dict[label][PID] = [pid]
            
    for label, pid_values in pid_dict.items():
        # 장수 
        len_imgs = [len(imgs) for imgs in pid_values.values()]
        print(f"Label {label} : {sum(len_imgs)}장 / {len(pid_values)}명")
    
    return pid_dict['0'], pid_dict['1'], pid_dict['2']


def _make_data_df(label_0, label_1, label_2):
    df = pd.DataFrame([], columns =['label', 'pid', 'img_name'])
    for label, label_dict in zip(['0', '1', '2'], [label_0, label_1, label_2]):
        for pid, imgs in label_dict.items():
            for img in imgs:
                df.loc[len(df)] = [label, pid, img]
    return df

def kfold_extract(csv_path, n_splits = 5, shuffle = True, random_state = 627):
    df = pd.read_csv(csv_path)

    #  data split 
    folds, train_df, test_df = ms_kfold(df = df, n_splits= n_splits, random_state= random_state, shuffle= shuffle)
    # 검증 함수 호출
    validate_pid_distributions(folds, train_df, test_df)
    # 레이블 분포 확인 함수 호출
    # print_label_distribution(folds, train_df, test_df)
    return folds, train_df, test_df

def label_counter(loader):
    label_counter = {
        '0': 0,
        '1': 0,
        '2': 0,
    }

    for inputs, masks, labels in loader:
        # label 클래스별 개수 확인
        label_counter['0'] += labels.numpy().tolist().count(0)  # 쉼표 제거
        label_counter['1'] += labels.numpy().tolist().count(1)  # 쉼표 제거
        label_counter['2'] += labels.numpy().tolist().count(2)  # 쉼표 제거
        
    return label_counter

