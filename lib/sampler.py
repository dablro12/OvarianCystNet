import torch
is_torchvision_installed = True
try:
    import torchvision
except:
    is_torchvision_installed = False
import torch.utils.data
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# 2. 각 pid의 다수 레이블을 계산
def multilabel_stratified_kfold(df: pd.DataFrame, n_splits: int, random_state: int = 627, shuffle: bool = True):
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
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




#%% BalancedBatchSampler
# class BalancedBatchSampler(torch.utils.data.Sampler):
#     def __init__(self, dataset, labels=None, batch_size=4):
#         self.labels = labels
#         self.dataset = dict()
#         self.balanced_max = 0
#         self.batch_size = batch_size

#         # 클래스별 인덱스 저장
#         for idx in range(len(dataset)):
#             label = self._get_label(dataset, idx)
#             if label not in self.dataset:
#                 self.dataset[label] = []
#             self.dataset[label].append(idx)
#             self.balanced_max = max(len(self.dataset[label]), self.balanced_max)

#         # 클래스별 인덱스 오버샘플링
#         for label in self.dataset:
#             while len(self.dataset[label]) < self.balanced_max:
#                 self.dataset[label].append(random.choice(self.dataset[label]))

#         self.keys = list(self.dataset.keys())
#         self.num_classes = len(self.keys)
#         if self.batch_size % self.num_classes != 0:
#             raise ValueError("Batch size must be divisible by number of classes")
        
#         self.samples_per_class = self.batch_size // self.num_classes

#     def __iter__(self):
#         # 각 클래스의 인덱스 섞기
#         shuffled_dataset = {label: random.sample(indices, len(indices)) for label, indices in self.dataset.items()}

#         # 총 배치 수 계산
#         num_batches = self.balanced_max // self.samples_per_class

#         # 모든 배치를 저장할 리스트
#         all_batches = []

#         for i in range(num_batches):
#             batch = []
#             for label in self.keys:
#                 start = i * self.samples_per_class
#                 end = (i + 1) * self.samples_per_class
#                 batch.extend(shuffled_dataset[label][start:end])
#             # 배치 내 샘플 순서 섞기
#             random.shuffle(batch)
#             all_batches.append(batch)

#         # 전체 배치 순서 섞기
#         random.shuffle(all_batches)

#         # 인덱스를 순차적으로 yield
#         for batch in all_batches:
#             for idx in batch:
#                 yield idx

#     def _get_label(self, dataset, idx):
#         if self.labels is not None:
#             return self.labels[idx].item()
#         else:
#             # 기본 라벨 추출 방식
#             dataset_type = type(dataset)
#             if 'torchvision' in dataset_type.__module__:
#                 if isinstance(dataset, torch.utils.data.Dataset):
#                     if hasattr(dataset, 'targets'):
#                         return dataset.targets[idx]
#                     elif hasattr(dataset, 'imgs'):
#                         return dataset.imgs[idx][1]
#             raise Exception("You should pass the tensor of labels to the constructor as second argument")

#     def __len__(self):
#         return self.balanced_max * self.num_classes

import torch
import random
import math

class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, labels=None, batch_size=4):
        """
        균형 잡힌 배치를 생성하기 위한 샘플러.

        Args:
            dataset (Dataset): PyTorch Dataset 객체.
            labels (Tensor, optional): 레이블 텐서. 제공되지 않으면 데이터셋에서 레이블을 추출.
            batch_size (int): 배치 크기. 클래스 수로 나누어 떨어져야 함.
        """
        self.labels = labels
        self.dataset = dict()
        self.batch_size = batch_size

        # 클래스별 인덱스 저장
        for idx in range(len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = []
            self.dataset[label].append(idx)

        self.keys = list(self.dataset.keys())
        self.num_classes = len(self.keys)

        if self.batch_size % self.num_classes != 0:
            raise ValueError("Batch size must be divisible by number of classes")

        self.samples_per_class = self.batch_size // self.num_classes

        # 각 클래스의 샘플 수 중 최대값 찾기
        original_max = max(len(indices) for indices in self.dataset.values())
        # balanced_max를 samples_per_class의 최소 배수로 설정
        self.balanced_max = math.ceil(original_max / self.samples_per_class) * self.samples_per_class

        # 클래스별 인덱스 오버샘플링
        for label in self.dataset:
            current_len = len(self.dataset[label])
            if current_len < self.balanced_max:
                num_to_add = self.balanced_max - current_len
                # 무작위로 샘플 복제
                self.dataset[label].extend(random.choices(self.dataset[label], k=num_to_add))

        # 모든 클래스가 balanced_max 개의 샘플을 가지는지 확인
        for label, indices in self.dataset.items():
            assert len(indices) == self.balanced_max, f"Class {label} has {len(indices)} samples, expected {self.balanced_max}"

    def __iter__(self):
        """
        샘플러의 이터레이터.

        Returns:
            Iterator of indices.
        """
        # 각 클래스의 인덱스 무작위로 섞기
        shuffled_dataset = {label: random.sample(indices, len(indices)) for label, indices in self.dataset.items()}

        # 총 배치 수 계산
        num_batches = self.balanced_max // self.samples_per_class

        all_batches = []

        for i in range(num_batches):
            batch = []
            for label in self.keys:
                start = i * self.samples_per_class
                end = (i + 1) * self.samples_per_class
                batch.extend(shuffled_dataset[label][start:end])
            random.shuffle(batch)  # 배치 내 샘플 순서 섞기
            all_batches.append(batch)

        random.shuffle(all_batches)  # 전체 배치 순서 섞기

        # 인덱스를 순차적으로 yield
        for batch in all_batches:
            for idx in batch:
                yield idx

    def _get_label(self, dataset, idx):
        """
        주어진 인덱스의 레이블을 반환.

        Args:
            dataset (Dataset): PyTorch Dataset 객체.
            idx (int): 데이터 인덱스.

        Returns:
            레이블 값.
        """
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            # 기본 레이블 추출 방식 (사용자 정의 데이터셋에 맞게 수정 필요)
            if hasattr(dataset, 'labels'):
                return dataset.labels[idx]
            elif hasattr(dataset, 'targets'):
                return dataset.targets[idx]
            elif hasattr(dataset, 'imgs'):
                return dataset.imgs[idx][1]
            else:
                raise AttributeError("Dataset에 레이블을 추출할 수 있는 속성이 없습니다. labels 인자를 제공하세요.")

    def __len__(self):
        """
        샘플러가 생성할 총 샘플 수.

        Returns:
            int: 총 샘플 수.
        """
        return self.balanced_max * self.num_classes
