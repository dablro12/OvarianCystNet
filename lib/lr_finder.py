#!/usr/bin/env python
import sys
sys.path.append('../')
import os
import gc
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch_lr_finder import LRFinder
import matplotlib.pyplot as plt
from lib.seed import set_seed
from lib.dataset import Custom_pcos_dataset, JointTransform, get_class_weights, compute_mean_std
from lib.sampler import BalancedBatchSampler
from model.loader import model_Loader
from lib.metric.metrics import multi_classify_metrics_v2

# 로깅 설정 (선택 사항)
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)

def main():
    # 시드 설정
    set_seed(42)
    
    # 데이터 경로 설정
    data_dir = '/mnt/hdd/octc/PCOS_Dataset'
    train_csv_path = os.path.join(data_dir, 'train.csv')
    valid_csv_path = os.path.join(data_dir, 'test.csv')
    
    # 데이터프레임 로드
    train_df = pd.read_csv(train_csv_path)
    valid_df = pd.read_csv(valid_csv_path)
    
    # 전체 데이터프레임 결합 (평균과 표준편차 계산을 위해)
    total_df = pd.concat([train_df, valid_df], axis=0)
    
    # 평균과 표준편차 계산
    mean, std = compute_mean_std(
        dataset=Custom_pcos_dataset(
            df=total_df,
            root_dir=data_dir,
            joint_transform=None,
            torch_transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]),
            mask_use=False,
            class_num=3,  # 클래스 수를 통일 (예: 3개)
            data_type="Dataset"
        )
    )
    print("Computed Mean:", mean)
    print("Computed Std:", std)
    
    # 데이터 증강 및 변환 정의
    train_transform = JointTransform(
        resize=(284, 284),
        center_crop=(224, 224),
        horizontal_flip=True,   # 데이터 증강수 : x2
        vertical_flip=True,     # 데이터 증강수 : x2
        random_affine=True,
        rotation=45,           # 데이터 증강수 : x8
        random_brightness=True,
        normalize_mean = mean.tolist(),
        normalize_std = std.tolist()
    )
    
    valid_transform = JointTransform(
        resize=(224, 224),
        normalize_mean=mean.tolist(),
        normalize_std=std.tolist()
    )
    
    # 데이터셋 정의
    train_dataset = Custom_pcos_dataset(
        df=train_df,
        root_dir=data_dir,
        joint_transform=train_transform,
        mask_use=False,
        class_num=3,  # 클래스 수를 통일
        data_type="Dataset"
    )
    
    valid_dataset = Custom_pcos_dataset(
        df=valid_df,
        root_dir=data_dir,
        joint_transform=valid_transform,
        torch_transform=False,  # 검증 데이터에는 데이터 증강을 적용하지 않음
        mask_use=False,
        class_num=3,  # 클래스 수를 통일
        data_type="Dataset"
    )
    
    # 데이터로더 정의
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=BalancedBatchSampler(train_dataset, train_dataset.labels_tensor),
        num_workers=8,  # 시스템에 맞게 조정
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=16,
        shuffle=False,  # 검증 데이터는 섞지 않음
        num_workers=8,  # 시스템에 맞게 조정
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 클래스 분포 출력
    train_unique, train_cnts = np.unique(train_dataset.labels_tensor.numpy(), return_counts=True)
    valid_unique, valid_cnts = np.unique(valid_dataset.labels_tensor.numpy(), return_counts=True)
    print(f"[Train] : {dict(zip(train_unique, train_cnts))}")
    print(f"[Valid] : {dict(zip(valid_unique, valid_cnts))}")
    
    # 클래스 가중치 계산
    weight = get_class_weights(dataset=train_dataset)
    print("Class Weights:", weight)
    
    # 손실 함수 정의
    criterion = nn.CrossEntropyLoss(weight=weight).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 로드
    model = model_Loader(
        model_name="resnet",
        outlayer_num=3,  # 클래스 수에 맞게 출력층 조정
        type="resnet34"
    )
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 옵티마이저 정의
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # 학습률 찾기 실행
    # 학습률 찾기는 전체 데이터를 사용하지 않고 일부 데이터를 사용하여 실행하는 것이 효율적입니다.
    subset_size = 1000  # 서브셋 크기 설정
    train_subset, _ = torch.utils.data.random_split(train_dataset, [subset_size, len(train_dataset) - subset_size])
    train_loader_subset = DataLoader(
        train_subset,
        batch_size=32,
        sampler=BalancedBatchSampler(train_subset, train_subset.labels_tensor),
        num_workers=8,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Learning Rate Finder 초기화 및 실행
    lr_finder = LRFinder(model, optimizer, criterion, device='cuda' if torch.cuda.is_available() else 'cpu')
    lr_finder.range_test(train_loader_subset, end_lr=1, num_iter=100, step_mode="exp")
    lr_finder.plot(suggest=True)
    plt.show()
    
    # 추천된 학습률을 기반으로 옵티마이저의 학습률을 업데이트
    suggested_lr = lr_finder.suggest()
    print(f"Suggested Learning Rate: {suggested_lr}")
    lr_finder.reset()  # 모델과 옵티마이저 상태를 초기화
    
    # 옵티마이저의 학습률 업데이트
    for param_group in optimizer.param_groups:
        param_group['lr'] = suggested_lr
    
    print(f"Updated Learning Rate to: {suggested_lr}")
    
    # 학습 루프 정의
    num_epochs = 50
    best_val_loss = float('inf')
    early_stop_patience = 10
    early_stop_counter = 0
    
    for epoch in range(1, num_epochs + 1):
        # 훈련 단계
        model.train()
        total_train_loss = 0.0
        all_train_labels = []
        all_train_preds = []
        
        for inputs, _, labels in train_loader:
            inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(preds.cpu().numpy())
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_metrics = multi_classify_metrics_v2(
            y_true=np.array(all_train_labels),
            y_prob=np.array(all_train_preds),
            optimize_thresholds=False  # 학습 단계에서는 임계값 최적화 비활성화
        )
        
        print(f"Epoch [{epoch}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Train Acc: {train_metrics.get('acc', 0):.2f}%")
        
        # 검증 단계
        model.eval()
        total_val_loss = 0.0
        all_val_labels = []
        all_val_probs = []
        
        with torch.no_grad():
            for inputs, _, labels in valid_loader:
                inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
                labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_probs.extend(probs.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(valid_loader)
        val_metrics = multi_classify_metrics_v2(
            y_true=np.array(all_val_labels),
            y_prob=np.array(all_val_probs),
            optimize_thresholds=True  # 검증 단계에서는 임계값 최적화 활성화
        )
        
        print(f"Epoch [{epoch}/{num_epochs}] Val Loss: {avg_val_loss:.4f} | Val Acc: {val_metrics.get('acc_opt', 0):.2f}% | Val AUC: {val_metrics.get('auc_opt', 0):.4f}")
        
        # 조기 종료 및 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            # 최적 모델 저장
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, os.path.join(data_dir, 'best_model.pth'))
            print(f"Best model saved at epoch {epoch}")
        else:
            early_stop_counter += 1
            print(f"No improvement in validation loss for {early_stop_counter} epochs.")
            if early_stop_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break
        
        # 학습률 스케줄러가 있다면 여기서 업데이트 (예: ReduceLROnPlateau)
        # 예시로 손실 기반 학습률 스케줄러 사용
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        scheduler.step(avg_val_loss)
    
    # 학습 완료 후 메모리 정리
    gc.collect()
    torch.cuda.empty_cache()


if __name__=="__main__":
    main()