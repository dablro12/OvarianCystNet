#!/usr/bin/env python
import sys
import os
import gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import argparse
from typing import Tuple, List, Any, Dict

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from ptflops import get_model_complexity_info
from torchsampler import ImbalancedDatasetSampler

import wandb
from lib.schedulers import lr_lambda
from lib.seed import set_seed
from lib.loss import FocalLoss, smooth_labels
from lib.dataset import Custom_df_dataset, JointTransform, Custom_pcos_dataset, get_class_weights, compute_mean_std
from lib.datasets.sampler import class_weight_getter
from lib.datasets.ds_tools import label_counter
from model.loader import model_Loader
from lib.metric.metrics import multi_classify_metrics_v2, binary_classify_metrics_v2
from lib.ml.kfold import k_fold_split
from lib.sampler import BalancedBatchSampler
from torch_lr_finder import LRFinder
set_seed(42)

#%% Multi Classificaiton
# class multi_exp_classification:
    # def __init__(
    #     self, 
    #     args=None, 
    # ):

    #     self.args = args
    #     self.model_name = args.model_name
    #     self.model_type = args.model_type
    #     self.save_dir = args.save_dir
    #     self.patience = args.patience
        
    #     # 아래 속성들은 main에서 할당해주는 방식 (trainer.run_name 등)이거나
    #     # 직접 None으로 초기화하고 뒤에서 필요한 경우 할당
    #     self.run_name = args.run_name if args else None
    #     self.wandb_use = args.wandb_use if args else None

    #     self.device = self._init_device()

    # def _init_device(self):
    #     return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # def _get_ds_loader(self):
    #     print(f"\033[41m[Run Name] {self.run_name}\033[0m")
    #     mean, std = compute_mean_std(
    #         dataset = Custom_pcos_dataset(
    #             pd.read_csv('/mnt/hdd/octc/PCOS_Dataset/train_split.csv'),
    #             root_dir="/mnt/hdd/octc/PCOS_Dataset",
    #             joint_transform=False,
    #             torch_transform = transforms.Compose([
    #                 transforms.Resize((224, 224)),
    #                 transforms.ToTensor(),
    #             ]),
    #             mask_use = False,
    #             class_num = 3,
    #             data_type = "Dataset"
    #         )
    #     )
    #     print(f"mean: {np.round(mean.tolist(), 3)}, std: {np.round(std.tolist(),3)}")
    #     transform = JointTransform(
    #         # resize=(224, 224),
    #         resize=(284, 284),
    #         center_crop=(224, 224),
    #         horizontal_flip=True,   # 데이터 증강수 : x2
    #         vertical_flip=True,     # 데이터 증강수 : x2
    #         rotation=10,           # 데이터 증강수 : x8
    #         random_brightness=True,
    #         # normalize_mean = mean.tolist(),
    #         # normalize_std = std.tolist()
    #     )
    #     train_dataset = Custom_pcos_dataset(
    #         df=pd.read_csv("/mnt/hdd/octc/PCOS_Dataset/train_split.csv"),
    #         root_dir="/mnt/hdd/octc/PCOS_Dataset",
    #         joint_transform= transform,
    #         torch_transform= False,
    #         mask_use=False,
    #         class_num=3,
    #         data_type="Dataset",
    #     )
        
    #     val_dataset = Custom_pcos_dataset(
    #         df=pd.read_csv("/mnt/hdd/octc/PCOS_Dataset/test_split.csv"),
    #         root_dir="/mnt/hdd/octc/PCOS_Dataset",
    #         joint_transform=JointTransform(
    #             resize=(284, 284),
    #             center_crop = (224, 224),
    #             # normalize_mean = mean.tolist(),
    #             # normalize_std = std.tolist()
    #         ),
    #         torch_transform= False,
    #         mask_use=False,
    #         class_num=3,
    #         data_type="Dataset"
    #     )
        
    #     train_loader = DataLoader(
    #         train_dataset,
    #         batch_size=48,
    #         shuffle = True,
    #         sampler=BalancedBatchSampler(train_dataset, labels=train_dataset.labels_tensor), 
    #         num_workers=16,
    #         pin_memory=True if self.device.type == 'cuda' else False,
    #     )
    #     val_loader = DataLoader(
    #         val_dataset,
    #         batch_size=8,
    #         shuffle=False,
    #         num_workers=16,
    #         pin_memory=True if self.device.type == 'cuda' else False,
    #     )
        
    #     # 데이터 분포를 통해 Crossweight에 가중치를 부여
    #     self.weight_tensor = get_class_weights(dataset = train_dataset)
        
    #     print("\033[41mFinished Data Loading\033[0m")
    #     print("\033[41mDataset Distribution\033[0m")
    #     print("[Train Distribution]",label_counter(train_loader))
    #     print("[Valid Distribution]",label_counter(val_loader))
    #     print(f"Weight Tensor: {self.weight_tensor.tolist()}")
        
    #     return train_loader, val_loader
        
    # def _get_model_loader(self):
    #     model = model_Loader(model_name=self.model_name, outlayer_num=3, type=self.model_type)
    #     model.to(self.device)  # 모델을 디바이스로 이동
        
    #     optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=1e-4)
        
    #     criterion = nn.CrossEntropyLoss(weight = self.weight_tensor, label_smoothing= 0.1).to(self.device)  # 손실함수
    #     # criterion = FocalLoss(alpha=self.weight_tensor, gamma=2, reduction='mean').to(self.device)  # Focal Loss

    #     # scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
    #     # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    #     # scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_up=5, step_size_down=5, mode='triangular2')
    #     scheduler = None
    #     print("\033[41mFinished Model Initialization\033[0m")
    #     return model, optimizer, criterion, scheduler
    
    # def save_best_model(self, epoch: int, val_loss: float):
    #     try:
    #         if self.best_model_path and os.path.exists(self.best_model_path):
    #             os.remove(self.best_model_path)
    #             print(f"이전 최적 모델을 삭제했습니다: {self.best_model_path}")

    #         # 저장 디렉토리 생성
    #         save_dir = os.path.join(self.args.save_dir, self.model_name+'_'+self.run_name)
    #         os.makedirs(save_dir, exist_ok=True)
    #         save_path = os.path.join(save_dir, f"best_model_fold_0_epoch_{epoch}.pth")

    #         torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': self.model.state_dict(),
    #             'loss': val_loss,
    #         }, save_path)
    #         print(f"최적 모델이 다음 경로에 저장되었습니다: {save_path}")

    #         self.best_model_path = save_path
    #     except Exception as e:
    #         print(f"에포크 {epoch}에서 모델 저장 중 오류 발생: {e}")
            
    # def validate(self, epoch: int) -> float:
    #     # 검증 함수 수정
    #     self.model.eval()
    #     total_loss = 0.0

    #     all_labels = []
    #     all_probs = []

    #     with torch.no_grad():
    #         for inputs, _, labels in self.val_loader:
    #             inputs = inputs.to(self.device)
    #             labels = labels.to(self.device).float()

    #             outputs = self.model(inputs)

    #             # 다중 분류
    #             _, predicted = torch.max(outputs.data, 1)
    #             labels = labels.long()

    #             loss = self.criterion(outputs, labels)
    #             total_loss += loss.item()

    #             all_labels.extend(labels.detach().cpu().numpy())
    #             all_probs.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())

    #     valid_loss = total_loss / len(self.val_loader)

    #     if not np.isfinite(valid_loss):
    #         print(f"경고: 에포크 {epoch}에서 유효하지 않은 평균 손실이 감지되었습니다. 손실을 무한대로 설정합니다.")
    #         valid_loss = float('inf')

    #     # 검증 단계에서는 임계값 최적화 활성화
    #     metrics = multi_classify_metrics_v2(
    #         y_true=np.array(all_labels),
    #         y_prob=np.array(all_probs),
    #         optimize_thresholds=True  # 임계값 최적화 활성화
    #     )
    #     metrics['valid_loss'] = valid_loss

    #     # -- (1) 원하는 메트릭을 WandB에 로그로 남길 수 있음
    #     if self.wandb_use:
    #         wandb.log({
    #             "valid_loss": metrics.get("valid_loss", 0),
    #             "valid_thresholds_Normal": metrics.get("thresholds", 0)[0],
    #             "valid_thresholds_Borderline": metrics.get("thresholds", 0)[1],
    #             "valid_thresholds_Abnormal": metrics.get("thresholds", 0)[2],
    #             "valid_acc": metrics.get("acc", 0),      
    #             "valid_acc_opt": metrics.get("acc_opt", 0),      
    #             "valid_f1": metrics.get("f1", 0),
    #             "valid_f1_opt": metrics.get("f1_opt", 0),
    #             "valid_auc": metrics.get("auc", 0),
    #             "valid_auc_opt": metrics.get("auc_opt", 0),
    #             "valid_specificity_Normal" : metrics.get("specificity", 0)[0],
    #             "valid_specificity_Borderline" : metrics.get("specificity", 0)[1],
    #             "valid_specificity_Abnormal" : metrics.get("specificity", 0)[2],
    #             "valid_sensitivity_Normal" : metrics.get("sensitivity", 0)[0],
    #             "valid_sensitivity_Borderline" : metrics.get("sensitivity", 0)[1],
    #             "valid_sensitivity_Abnormal" : metrics.get("sensitivity", 0)[2],
    #             "valid_specificity_Normal_opt" : metrics.get("specificity_opt", 0)[0],
    #             "valid_specificity_Borderline_opt" : metrics.get("specificity_opt", 0)[1],
    #             "valid_specificity_Abnormal_opt" : metrics.get("specificity_opt", 0)[2],
    #             "valid_sensitivity_Normal_opt" : metrics.get("sensitivity_opt", 0)[0],
    #             "valid_sensitivity_Borderline_opt" : metrics.get("sensitivity_opt", 0)[1],
    #             "valid_sensitivity_Abnormal_opt" : metrics.get("sensitivity_opt", 0)[2],
    #         })

    #     print(f"[Validate] Epoch={epoch}, Loss={valid_loss:.4f}")
    #     print(f'정확도={metrics.get("acc_opt",0)}%\nAUROC={metrics.get("auc_opt", 0)}\nF1={metrics.get("f1_opt", 0)}%\n재현율={metrics.get("sensitivity_opt", 0)}%\n특이도={metrics.get("specificity_opt", 0)}%')

    #     return valid_loss

    # def train_epoch(self, epochs: int) -> Tuple[float, int]:
    #     final_val_loss = float('inf')

    #     for epoch in range(1, epochs+1):
    #         self.model.train()
    #         total_loss = 0.0
            
    #         all_labels = []
    #         all_probs = []
            
    #         for inputs, _, labels in self.train_loader:
    #             self.optimizer.zero_grad()
    #             inputs = inputs.to(self.device, non_blocking=True)
    #             labels = labels.to(self.device, non_blocking=True).float()

    #             outputs = self.model(inputs)
    #             _, predicted = torch.max(outputs.data, 1)
    #             labels = labels.long()
                    
    #             loss = self.criterion(outputs, labels)
    #             loss.backward()
    #             self.optimizer.step()
    #             total_loss += loss.item()

    #             all_labels.extend(labels.detach().cpu().numpy())
    #             all_probs.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())

    #         train_loss = total_loss / len(self.train_loader)
            
    #         metrics = multi_classify_metrics_v2(
    #             y_true = np.array(all_labels),
    #             y_prob = np.array(all_probs),
    #             optimize_thresholds=True  # 임계값 최적화 활성화
    #         )
    #         print(f"[Train] Epoch [{epoch}/{epochs}], Loss: {train_loss:.4f}")
    #         print(f'정확도={metrics.get("acc_opt",0)}%\nAUROC={metrics.get("auc_opt", 0)}\nF1={metrics.get("f1_opt", 0)}%\n재현율={metrics.get("sensitivity_opt", 0)}%\n특이도={metrics.get("specificity_opt", 0)}%')

    #         # -- (2) train-loss, train-accuracy를 WandB에 로깅
    #         if self.wandb_use:
    #             wandb.log({
    #                 "learning_rate": self.optimizer.param_groups[0]['lr'],
    #                 "train_thresholds_Normal": metrics.get("thresholds", 0)[0],
    #                 "train_thresholds_Borderline": metrics.get("thresholds", 0)[1],
    #                 "train_thresholds_Abnormal": metrics.get("thresholds", 0)[2],
    #                 "train_loss": train_loss,
    #                 "train_acc": metrics.get("acc", 0),      # multi_classify_metrics_v2 내에서 키를 "accuracy"로 사용
    #                 "train_acc_opt": metrics.get("acc_opt", 0),      # multi_classify_metrics_v2 내에서 키를 "accuracy"로 사용
    #                 "train_f1": metrics.get("f1", 0),
    #                 "train_f1_opt": metrics.get("f1_opt", 0),
    #                 "train_auc": metrics.get("auc", 0),
    #                 "train_auc_opt": metrics.get("auc_opt", 0),
    #                 "train_specificity_Normal" : metrics.get("specificity", 0)[0],
    #                 "train_specificity_Borderline" : metrics.get("specificity", 0)[1],
    #                 "train_specificity_Abnormal" : metrics.get("specificity", 0)[2],
    #                 "train_sensitivity_Normal" : metrics.get("sensitivity", 0)[0],
    #                 "train_sensitivity_Borderline" : metrics.get("sensitivity", 0)[1],
    #                 "train_sensitivity_Abnormal" : metrics.get("sensitivity", 0)[2],
    #                 "train_specificity_Normal_opt" : metrics.get("specificity_opt", 0)[0],
    #                 "train_specificity_Borderline_opt" : metrics.get("specificity_opt", 0)[1],
    #                 "train_specificity_Abnormal_opt" : metrics.get("specificity_opt", 0)[2],
    #                 "train_sensitivity_Normal_opt" : metrics.get("sensitivity_opt", 0)[0],
    #                 "train_sensitivity_Borderline_opt" : metrics.get("sensitivity_opt", 0)[1],
    #                 "train_sensitivity_Abnormal_opt" : metrics.get("sensitivity_opt", 0)[2],
    #             })

    #         # Validation
    #         val_loss = self.validate(epoch=epoch)
    #         final_val_loss = val_loss

    #         # 학습률 스케줄러
    #         self.scheduler.step(val_loss)

    #         # Early stopping 로직
    #         if val_loss < self.best_val_loss:
    #             self.best_val_loss = val_loss
    #             self.early_stop_counter = 0
    #             self.save_best_model(epoch, val_loss)
    #         else:
    #             self.early_stop_counter += 1

    #         if self.early_stop_counter >= self.patience:
    #             print("Early stopping triggered")
    #             self.save_best_model(epoch, val_loss)
    #             break

    #     return final_val_loss, epoch
        
    # def fit(self, epochs: int):
    #     set_seed(42)
    #     self.best_val_loss = float('inf')
    #     self.early_stop_counter = 0
    #     self.best_model_path = None
        
    #     # -- (3) WandB 초기화 (선택적으로 run_name, project, entity 등 지정 가능)
    #     if self.wandb_use:
    #         wandb.init(
    #             project="multi-octc-classifier",  # 원하는 프로젝트 명
    #             name=self.model_name + "_" + self.run_name,
    #         )
        
    #     self.train_loader, self.val_loader = self._get_ds_loader()
    #     self.model, self.optimizer, self.criterion, self.scheduler = self._get_model_loader()
        
    #     val_loss, best_epoch = self.train_epoch(epochs=epochs)
        
    #     # 모델이 저장되지 않았다면 마지막 에포크의 모델을 저장
    #     if self.best_model_path is None and best_epoch != -1:
    #         print("학습 중 개선이 감지되지 않았습니다. 마지막 에포크의 모델을 저장합니다.")
    #         self.save_best_model(best_epoch, val_loss)

    #     # -- (4) WandB 세션 종료
    #     if self.wandb_use:
    #         wandb.finish()

    #     # 가비지 컬렉션
    #     gc.collect()
    #     torch.cuda.empty_cache()
#%% Binary Classificaiton
class binary_exp_classification:
    def __init__(
        self, 
        args=None, 
    ):
        self.args = args
        self.model_name = args.model_name
        self.model_type = args.model_type
        self.save_dir = args.save_dir
        self.patience = args.patience
        
        # 아래 속성들은 main에서 할당해주는 방식 (trainer.run_name 등)이거나
        # 직접 None으로 초기화하고 뒤에서 필요한 경우 할당
        self.run_name = args.run_name if args else None
        self.wandb_use = args.wandb_use if args else None

        self.device = self._init_device()
        print(f"\033[41m[Task] Binary Classification\033[0m")

    def _init_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _get_ds_loader(self):
        print(f"\033[41m[Run Name] {self.run_name}\033[0m")
        # mean, std = compute_mean_std(
        #     dataset = Custom_pcos_dataset(
        #         pd.read_csv('/mnt/hdd/octc/PCOS_Dataset/train_split.csv'),
        #         root_dir="/mnt/hdd/octc/PCOS_Dataset",
        #         joint_transform=False,
        #         torch_transform = transforms.Compose([
        #             transforms.Resize((224, 224)),
        #             transforms.ToTensor(),
        #         ]),
        #         mask_use = False,
        #         class_num = 1,  # 3에서 1로 변경 (이진 분류)
        #         data_type = "Dataset"
        #     )
        # )
        # print(f"mean: {np.round(mean.tolist(), 3)}, std: {np.round(std.tolist(),3)}")
        train_transform = JointTransform(
            # #%% Original
            # resize =(224,224),
            # #%% Aug
            resize =(284,284),
            center_crop=(224, 224),
            horizontal_flip=True,   # 데이터 증강: x2
            vertical_flip=True,     # 데이터 증강: x2
            # random_affine=True,
            rotation=10,            # 데이터 증강: x8
            random_brightness=True,
            
            # Default
            # normalize_mean = mean.tolist(),
            # normalize_std = std.tolist()
        )
        
        train_dataset = Custom_pcos_dataset(
            df=pd.read_csv("/mnt/hdd/octc/PCOS_Dataset/augcombined_train_split.csv"),
            root_dir="/mnt/hdd/octc/PCOS_Dataset",
            joint_transform= train_transform,
            torch_transform= False,
            mask_use=False,
            class_num=1,  # 3에서 1로 변경
            data_type="Dataset-combined-AUGGAN",
        )
        
        val_dataset = Custom_pcos_dataset(
            df=pd.read_csv("/mnt/hdd/octc/PCOS_Dataset/test_split.csv"),
            root_dir="/mnt/hdd/octc/PCOS_Dataset",
            joint_transform=JointTransform(
                resize=(284, 284),
                center_crop=(224, 224),
                # normalize_mean = mean.tolist(),
                # normalize_std = std.tolist()
            ),
            torch_transform= False,
            mask_use=False,
            class_num=1,  # 3에서 1로 변경
            data_type="Dataset"
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            # shuffle = True,
            sampler=BalancedBatchSampler(dataset = train_dataset, labels=train_dataset.labels_tensor), 
            num_workers=16,
            pin_memory=True if self.device.type == 'cuda' else False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=16,
            pin_memory=True if self.device.type == 'cuda' else False,
        )
        
        # 데이터 분포를 통해 클래스 가중치 계산
        self.weight_tensor = get_class_weights(dataset = train_dataset) # 음성 클래스 가중치 / 양성 클래스 가중치 * 역수임!
        
        print("\033[41mFinished Data Loading\033[0m")
        print("\033[41mDataset Distribution\033[0m")
        print("[Train Distribution]", label_counter(train_loader))
        print("[Valid Distribution]", label_counter(val_loader))
        print(f"Weight Tensor: {self.weight_tensor.tolist()}")
        
        return train_loader, val_loader
        
    def _get_model_loader(self):
        # 이진 분류에 맞게 출력 레이어 조정 (outlayer_num=1)
        model = model_Loader(model_name=self.model_name, outlayer_num=1, type=self.model_type)
        model.to(self.device)  # 모델을 디바이스로 이동
        
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
        
        # criterion = nn.BCEWithLogitsLoss().to(self.device)
        binary_pos_weight = self.weight_tensor[1] / self.weight_tensor[0] # pos_weight = 음성 클래스 수 / 양성 클래스 수
        # print(f"[양성 클래스 가중치] {binary_pos_weight}")
        criterion = nn.BCEWithLogitsLoss(pos_weight = binary_pos_weight).to(self.device)
        # criterion = nn.BCEWithLogitsLoss().to(self.device)
        # 학습률 스케줄러 설정
        # scheduler = None
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda = lr_lambda)
        print("\033[41mFinished Model Initialization\033[0m")
        return model, optimizer, criterion, scheduler
    
    def save_best_model(self, epoch: int, val_loss: float):
        try:
            if self.best_model_path and os.path.exists(self.best_model_path):
                os.remove(self.best_model_path)
                print(f"이전 최적 모델을 삭제했습니다: {self.best_model_path}")

            # 저장 디렉토리 생성
            save_dir = os.path.join(self.save_dir, self.model_name + '_' + self.run_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"best_model_epoch_{epoch}.pth")

            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'loss': val_loss,
            }, save_path)
            print(f"최적 모델이 다음 경로에 저장되었습니다: {save_path}")

            self.best_model_path = save_path
        except Exception as e:
            print(f"에포크 {epoch}에서 모델 저장 중 오류 발생: {e}")
            
    def validate(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0

        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, _, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).float()

                outputs = self.model(inputs)
                # 이진 분류: 시그모이드 활성화 적용하여 확률 계산
                probs = torch.sigmoid(outputs)
                
                # smoothed_labels = smooth_labels(labels, epsilon=0.1)
                # loss = self.criterion(outputs, smoothed_labels)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                all_labels.extend(labels.detach().cpu().numpy())
                all_probs.extend(probs.detach().cpu().numpy())

        valid_loss = total_loss / len(self.val_loader)

        if not np.isfinite(valid_loss):
            print(f"경고: 에포크 {epoch}에서 유효하지 않은 평균 손실이 감지되었습니다. 손실을 무한대로 설정합니다.")
            valid_loss = float('inf')

        # 이진 분류 메트릭 계산
        metrics = binary_classify_metrics_v2(
            y_true=np.array(all_labels),
            y_prob=np.array(all_probs),
            optimize_thresholds=True  # 임계값 최적화 활성화
        )
        metrics['valid_loss'] = valid_loss

        # 메트릭을 WandB에 로그
        if self.wandb_use:
            wandb.log({
                "valid_loss": metrics.get("valid_loss", 0),
                "valid_threshold_opt": metrics.get("threshold_opt", 0),
                "valid_acc": metrics.get("acc", 0),      
                "valid_acc_opt": metrics.get("acc_opt", 0),      
                "valid_f1": metrics.get("f1", 0),
                "valid_f1_opt": metrics.get("f1_opt", 0),
                "valid_auc": metrics.get("auc", 0),
                "valid_auc_opt": metrics.get("auc_opt", 0),
                "valid_specificity": metrics.get("specificity", 0),
                "valid_sensitivity": metrics.get("sensitivity", 0),
                "valid_specificity_opt": metrics.get("specificity_opt", 0),
                "valid_sensitivity_opt": metrics.get("sensitivity_opt", 0),
            })

        print(f"[Validate] Epoch={epoch}, Loss={valid_loss:.4f}")
        print(f'정확도={metrics.get("acc_opt",0)}%\nAUROC={metrics.get("auc_opt", 0)}\nF1={metrics.get("f1_opt", 0)}%\n재현율={metrics.get("sensitivity_opt", 0)}%\n특이도={metrics.get("specificity_opt", 0)}%')

        return valid_loss

    def train_epoch(self, epochs: int) -> Tuple[float, int]:
        final_val_loss = float('inf')

        for epoch in range(1, epochs+1):
            self.model.train()
            total_loss = 0.0
            
            all_labels = []
            all_probs = []
            
            for inputs, _, labels in self.train_loader:
                self.optimizer.zero_grad()
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).float()

                outputs = self.model(inputs)
                # 이진 분류: 시그모이드 활성화 적용하여 확률 계산
                probs = torch.sigmoid(outputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                all_labels.extend(labels.detach().cpu().numpy())
                all_probs.extend(probs.detach().cpu().numpy())

            train_loss = total_loss / len(self.train_loader)
            
            # 이진 분류 메트릭 계산
            metrics = binary_classify_metrics_v2(
                y_true = np.array(all_labels),
                y_prob = np.array(all_probs),
                optimize_thresholds=True  # 임계값 최적화 활성화
            )
            print(f"[Train] Epoch [{epoch}/{epochs}], Loss: {train_loss:.4f}")
            print(f'정확도={metrics.get("acc_opt",0)}%\nAUROC={metrics.get("auc_opt", 0)}\nF1={metrics.get("f1_opt", 0)}%\n재현율={metrics.get("sensitivity_opt", 0)}%\n특이도={metrics.get("specificity_opt", 0)}%')

            # 메트릭을 WandB에 로그
            if self.wandb_use:
                wandb.log({
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                    "train_threshold_opt": metrics.get("threshold_opt", 0),
                    "train_loss": train_loss,
                    "train_acc": metrics.get("acc", 0),      
                    "train_acc_opt": metrics.get("acc_opt", 0),      
                    "train_f1": metrics.get("f1", 0),
                    "train_f1_opt": metrics.get("f1_opt", 0),
                    "train_auc": metrics.get("auc", 0),
                    "train_auc_opt": metrics.get("auc_opt", 0),
                    "train_specificity": metrics.get("specificity", 0),
                    "train_sensitivity": metrics.get("sensitivity", 0),
                    "train_specificity_opt": metrics.get("specificity_opt", 0),
                    "train_sensitivity_opt": metrics.get("sensitivity_opt", 0),
                })

            # 검증
            val_loss = self.validate(epoch=epoch)
            final_val_loss = val_loss

            # # 학습률 스케줄러 단계
            # The above code is attempting to access the `scheduler` attribute of the `self` object in
            # a Python class.
            self.scheduler.step()

            # 조기 종료 로직
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                self.save_best_model(epoch, val_loss)
            else:
                self.early_stop_counter += 1

            if self.early_stop_counter >= self.patience:
                print("Early stopping triggered")
                self.save_best_model(epoch, val_loss)
                break

        return final_val_loss, epoch
    
    
            
    def fit(self, epochs: int):
        set_seed(42)
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        self.best_model_path = None
        
        # WandB 초기화
        if self.wandb_use:
            wandb.init(
                project="binary-octc-classifier",  # 프로젝트 명 업데이트
                name=self.model_name + "_" + self.run_name,
            )
        
        self.train_loader, self.val_loader = self._get_ds_loader()
        self.model, self.optimizer, self.criterion, self.scheduler = self._get_model_loader()
        
        val_loss, best_epoch = self.train_epoch(epochs=epochs)
        
        # 모델이 저장되지 않았다면 마지막 에포크의 모델을 저장
        if self.best_model_path is None and best_epoch != -1:
            print("학습 중 개선이 감지되지 않았습니다. 마지막 에포크의 모델을 저장합니다.")
            self.save_best_model(best_epoch, val_loss)

        # WandB 세션 종료
        if self.wandb_use:
            wandb.finish()

        # 가비지 컬렉션
        gc.collect()
        torch.cuda.empty_cache()
        
    