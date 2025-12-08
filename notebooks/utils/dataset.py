import pandas as pd

def create_label_mapping(label_df, label_col):
    unique_labels = sorted(label_df[label_col].unique())
    mapping = {label: idx for idx, label in enumerate(unique_labels)}
    print("Label Mapping:", mapping)
    return mapping

def normalize_label(label):
    # "PCOS,NORMAL" → 멀티라벨 → 리스트로 변환
    if isinstance(label, str) and "," in label:
        return label.split(",")
    return label

from sklearn.model_selection import train_test_split, StratifiedKFold


def stratified_split_by_pid(df, pid_col="pid", label_col="label", seed=42):
    # PID 기반 그룹핑
    pid_groups = df.groupby(pid_col)[label_col].agg(lambda x: x.iloc[0])
    pid_groups = pid_groups.reset_index()

    # Train 70%, Temp 30%
    train_pid, temp_pid = train_test_split(
        pid_groups,
        test_size=0.3,
        stratify=pid_groups[label_col],
        random_state=seed
    )

    # Temp을 다시 Val 10%, Test 20%로 나누기 (Temp = 30 → Val 10, Test 20)
    val_pid, test_pid = train_test_split(
        temp_pid,
        test_size=2/3,     # (20% / 30%) = 2/3
        stratify=temp_pid[label_col],
        random_state=seed
    )

    # PID로 원본 df 필터링
    train_df = df[df[pid_col].isin(train_pid[pid_col])]
    val_df = df[df[pid_col].isin(val_pid[pid_col])]
    test_df = df[df[pid_col].isin(test_pid[pid_col])]

    print("PID Split Result:")
    print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))

    return train_df, val_df, test_df

def stratified_pid_kfold(df, n_splits=5, pid_col="pid", label_col="label", seed=42):
    """
    Returns list of (train_df, val_df) for each fold.
    """

    # PID 단위로 label 대표값 생성
    pid_groups = df.groupby(pid_col)[label_col].first().reset_index()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []

    for fold_idx, (train_pid_index, val_pid_index) in enumerate(
        skf.split(pid_groups[pid_col], pid_groups[label_col])
    ):
        train_pids = pid_groups.iloc[train_pid_index][pid_col].values
        val_pids = pid_groups.iloc[val_pid_index][pid_col].values

        train_df = df[df[pid_col].isin(train_pids)]
        val_df = df[df[pid_col].isin(val_pids)]

        print(f"[Fold {fold_idx}] Train: {len(train_df)}, Val: {len(val_df)}")

        folds.append((train_df, val_df))

    return folds


import torch
from torch.utils.data import WeightedRandomSampler

def create_weighted_sampler(df, label_col):
    class_counts = df[label_col].value_counts()
    weight_map = 1.0 / class_counts
    weights = df[label_col].map(weight_map).values
    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return sampler


import os
from PIL import Image
from torch.utils.data import Dataset
class PCOSDataset(Dataset):
    def __init__(self, df, data_root, filename_col, label_col, label_mapping,
                 transform=None, multi_label=False):
        
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.filename_col = filename_col
        self.label_col = label_col
        self.transform = transform
        self.label_mapping = label_mapping
        self.multi_label = multi_label

    def __len__(self):
        return len(self.df)

    def load_image(self, path):
        if not os.path.exists(path):
            print(f"[Warning] Image not found: {path} — skipping.")
            return Image.new("RGB", (224, 224), color=(0, 0, 0))  # 빈 이미지 반환
        
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        filename = row[self.filename_col]
        img_path = os.path.join(self.data_root, filename + '.png')

        image = self.load_image(img_path)

        # augmentation 적용
        if self.transform:
            image = self.transform(image)

        # multi-label / multi-class 지원
        label = normalize_label(row[self.label_col])

        if self.multi_label:
            # multi-label: ['PCOS','NORMAL'] → multi-hot 벡터로 변환 등 확장 가능
            numeric_label = [self.label_mapping[l] for l in label]
        else:
            numeric_label = self.label_mapping[label]

        return image, numeric_label


class HFVisionDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, processor = None):
        self.base = base_dataset
        self.processor = processor

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, label = self.base[idx]

        # PIL or tensor → processor(pixel_values=...) 변환

        # 배치 차원 제거
        if self.processor:
            encoded = self.processor(images=image, return_tensors="pt")
            return {
                "pixel_values": encoded.pixel_values,                             # Trainer가 이 키를 사용
                "labels": torch.tensor(label, dtype=torch.long),   # 정수 레이블
            }
        else:
            return {
            "pixel_values": image,                             # Trainer가 이 키를 사용
            "labels": torch.tensor(label, dtype=torch.long),   # 정수 레이블
            }



import pandas as pd 
from utils.transform import get_transform, SpeckleNoise, AddGaussianNoise
from torch.utils.data import DataLoader
from torchvision import transforms

def default_run(data_root_dir, label_path):
    label_df = pd.read_csv(label_path)

    train_tf, val_tf = get_transform(
        train_transform=[
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05,0.05), scale=(0.95,1.05)),
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
            # AddGaussianNoise(std=0.02),
            # SpeckleNoise(noise_factor=0.1),
        ]
    )

    label_mapping = create_label_mapping(label_df, "label")

    # 2) Train / Val / Test split (PID 단위 stratified 7:1:2)
    train_df, val_df, test_df = stratified_split_by_pid(label_df)

    # 3) Tune = Train + Val (80%)
    tune_df = pd.concat([train_df, val_df]).reset_index(drop=True)

    # 4) K-Fold는 tune_df 에 대해서만 적용
    folds = stratified_pid_kfold(tune_df, n_splits=5) # [ADD] K-Fold Cross Validation 추가
    kfold_loaders = []

    for fold_idx, (train_df, val_df) in enumerate(folds):
        train_dataset = PCOSDataset(train_df, data_root_dir, "filename", "label",
                                    label_mapping, transform=train_tf)
        val_dataset   = PCOSDataset(val_df,   data_root_dir, "filename", "label",
                                    label_mapping, transform=val_tf)

        sampler = create_weighted_sampler(train_df, "label")

        train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=4)
        val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

        kfold_loaders.append((train_loader, val_loader))

        print(f"Fold {fold_idx} loader ready.")

    # Test loader는 따로 구성
    test_dataset = PCOSDataset(test_df, data_root_dir, "filename", "label", label_mapping, transform=val_tf)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    return kfold_loaders, test_loader