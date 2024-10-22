import pandas as pd
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np 
import torch 
# 2. 각 pid의 다수 레이블을 계산
def multilabel_stratified_kfold(df: pd.DataFrame, n_splits: int, random_state: int = 627, shuffle: bool = True):
    pid_label = df.groupby('pid')['label'].agg(lambda x: x.mode()[0]).reset_index()
    pid_label = pid_label.rename(columns={'label': 'pid_label'})

    # DataFrame에 병합
    df = df.merge(pid_label, on='pid')

    # 3. Train/Test Split (80% Train, 20% Test)
    unique_pids = pid_label['pid']
    unique_labels = pid_label['pid_label']

    train_pids, test_pids, _, _ = train_test_split(
        unique_pids,
        unique_labels,
        test_size=0.2,
        random_state=random_state,
        stratify=unique_labels
    )

    train_df = df[df['pid'].isin(train_pids)].reset_index(drop=True)
    test_df = df[df['pid'].isin(test_pids)].reset_index(drop=True)

    # 4. Stratified K-Fold Cross-Validation
    # 4.1. 레이블 인코딩
    pid_label_train = pid_label[pid_label['pid'].isin(train_pids)].reset_index(drop=True)
    pid_label_train['pid_label'] = pid_label_train['pid_label'].astype(str)  # 문자열로 변환

    # 원-핫 인코딩 수행
    pid_label_ohe = pd.get_dummies(pid_label_train['pid_label'])

    # 4.2. Multilabel Stratified K-Fold 초기화
    n_splits = 5
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=42)

    # 4.3. K-Fold 분할 생성
    folds = []

    for fold, (train_idx, val_idx) in enumerate(mskf.split(pid_label_ohe, pid_label_ohe)):
        # 현재 폴드의 훈련 및 검증 pids
        train_fold_pids = pid_label_train.iloc[train_idx]['pid']
        val_fold_pids = pid_label_train.iloc[val_idx]['pid']
        
        # 데이터 분할
        train_fold = train_df[train_df['pid'].isin(train_fold_pids)].reset_index(drop=True)
        val_fold = train_df[train_df['pid'].isin(val_fold_pids)].reset_index(drop=True)
        
        folds.append({
            'train': train_fold,
            'val': val_fold
        })
    
    # Train 
    # print(">>> Iteration Stratificiation K-Fold Overview >>>")
    # print(f"Train pids: {train_pids.nunique()}")
    # print(f"Train labels distribution:\n{train_df['pid_label'].value_counts(normalize=True)}")
    # # Test 
    # print(f"Test pids: {test_pids.nunique()}")
    # print(f"Test labels distribution:\n{test_df['pid_label'].value_counts(normalize=True)}")

    return folds, train_df, test_df 


def class_weight_sampler(fold):
    class_counts = np.bincount(fold['train']['label'])
    class_weights = 1. / class_counts
    sample_weights = class_weights[fold['train']['label']]
    
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    return sampler 



def class_weight_getter(fold):
    n_pos = (fold['train']['label']==1).sum()
    n_neg = (fold['train']['label']==0).sum()
    pos_weight_value = n_neg / n_pos
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to('cuda')
    return pos_weight