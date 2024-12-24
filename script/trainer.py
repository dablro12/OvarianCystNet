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

from lib.seed import set_seed
from lib.dataset import Custom_df_dataset, JointTransform, Custom_pcos_dataset, get_class_weights
from lib.datasets.sampler import class_weight_getter
from model.loader import model_Loader
from lib.metric.metrics import multi_classify_metrics, binary_classify_metrics
from lib.ml.kfold import k_fold_split
from lib.sampler import BalancedBatchSampler

# 환경 변수 설정 및 백엔드 최적화
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.autograd.set_detect_anomaly(True)

class BaseClassifier:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = self.setup_device()
        
        if self.args.mask_use == 'yes':
            self.args.mask_use = True
            print(f"Mask Use")
        elif self.args.mask_use == 'no':
            self.args.mask_use = False
            print(f"Mask Not Use")

        self.wandb_use = self.check_wandb_use()
        self.best_accuracy = 0.0
        self.best_model_path = None  # Best 모델 경로를 저장할 변수 추가
        self.save_every = math.ceil(self.args.epochs / 10)
        self.input_res = (3, 224, 224)

        # 조기 종료 파라미터
        self.patience = 20
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0

        print("=" * 100, "\033[41mStart Initialization\033[0m")
        self.fit()
        print("\033[41mFinished Initialization\033[0m")

    def check_wandb_use(self):
        if self.args.wandb_use == 'yes':
            print(f"\033[41mWandB Use\033[0m")
            return True
        else:
            print(f"\033[41mWandB Not Use\033[0m")
            return False
    
    def setup_device(self) -> torch.device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\033[41mCUDA Status : {device.type}\033[0m")
        return device

    def init_augmentation(self) -> Tuple[List[Any], List[Any]]:
        # For Augment
        if self.args.version in ["originwithaugment", "inpaintnwithaugment", "masknwithaugment"]:
            print(f"Augment Use : Yes")
            train_augment_list = JointTransform(
                resize=(284, 284),
                center_crop= (self.input_res[1], self.input_res[2]),
                horizontal_flip=True, #데이터 증강수 : x 2
                vertical_flip=True, # 데이터 증강수 : x 2
                random_affine = True,
                rotation=30, # 데이터 증강수 : x 8
                random_brightness= True, #
            )
        else:#Form Original
            print(f"Augment Use : No")
            train_augment_list = JointTransform(
                resize=(self.input_res[1], self.input_res[2]),
            )

        valid_augment_list = JointTransform(
            resize=(self.input_res[1], self.input_res[2]),
        )
        return train_augment_list, valid_augment_list

    def init_dataset(self, fold) -> Tuple[DataLoader, DataLoader]:
        if 'inpaint' in str(self.args.version): # self.args.version안에 "inpaint"라는 단어가 있으면 데이터셋 종류를 변경
            data_type = 'Dataset_OCI'
        else:
            data_type = 'Dataset'
        print(f"Data type: {data_type}")
        train_dataset = Custom_pcos_dataset(
            df=fold['train'],
            root_dir=self.args.data_dir,
            joint_transform=self.args.train_augment_list,
            mask_use = self.args.mask_use,
            class_num = self.args.outlayer_num,
            data_type = data_type
        )
        
        val_dataset = Custom_pcos_dataset(
            df=fold['val'],
            root_dir=self.args.data_dir,
            joint_transform=self.args.valid_augment_list,
            mask_use = self.args.mask_use,
            class_num = self.args.outlayer_num,
            data_type = data_type
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=BalancedBatchSampler(train_dataset, labels = train_dataset.labels_tensor), # 변경: ImbalancedDatasetSampler -> BalancedBatchSampler
            num_workers=16,
            pin_memory=True if self.device.type == 'cuda' else False,
            # drop_last=False  # 변경: drop_last=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.valid_batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True if self.device.type == 'cuda' else False,
            # drop_last=False  # 변경: drop_last=False
        )

        return train_loader, val_loader
    
    def init_model(self, model_name: str, learning_rate: float, outlayer_num: int, fold) -> Tuple[nn.Module, optim.Optimizer, nn.Module, optim.lr_scheduler._LRScheduler]:
        model = model_Loader(model_name=model_name, outlayer_num=outlayer_num, type=self.args.type)
        model.to(self.device)  # 모델을 디바이스로 이동

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-4) # 파라미터 프리즈 후 옵티마이저 설정

        if outlayer_num > 1:
            class_weights = get_class_weights(fold['train']['label'].values, outlayer_num, device = self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        elif outlayer_num == 1:
            # 이진 분류의 경우 pos_weight 계산
            pos_weight = class_weight_getter(fold)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            raise ValueError("outlayer_num must be greater than 0.")

        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

        print("\033[41mFinished Model Initialization\033[0m")
        return model, optimizer, criterion, scheduler

    def calculate_model_params(self, model: nn.Module):
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n\033[0mTotal trainable parameters: {total_params}\033[0m")

        try:
            macs, params = get_model_complexity_info(model, self.input_res, as_strings=False,
                                                    print_per_layer_stat=False, verbose=False)
            flops = 2 * macs
            print(f"FLOPs: {flops}")
        except Exception as e:
            print(f"Error calculating FLOPs: {e}")
            flops = None

        self.args.total_params = total_params
        self.args.flops = flops

    def wandb_init(self):
        if self.wandb_use:
            print("\033[41mWandB Initialization\033[0m")
            wandb.init(
                project=self.args.wandb_project,
                config=self.args.__dict__
            )
            wandb.run.name = self.run_name
            wandb.watch(self.model, log="all", log_freq=100)
            print("\033[41mWandB Initialized\033[0m")
        else:
            print("\033[41mWandB Not Used\033[0m")

    def fit(self):
        fold_scores = []
        set_seed(self.args.seed)

        # 데이터 증강 초기화 (반복문 바깥에서 한 번만 호출)
        self.args.train_augment_list, self.args.valid_augment_list = self.init_augmentation()

        # K-Fold 데이터셋 초기화
        self.kfolds = k_fold_split(csv_path=self.args.csv_path, random_seed=self.args.seed)

        for idx, fold in enumerate(self.kfolds):
            # 각 폴드마다 조기 종료 변수 초기화
            self.best_val_loss = float('inf')
            self.early_stop_counter = 0
            self.best_model_path = None  # 폴드마다 초기화

            current_fold = idx + 1  # 현재 폴드 번호
            self.run_name = f"{self.args.backbone_model}_{self.args.type}_fold_{current_fold}-{self.args.version}"

            # 데이터 로더 초기화
            self.train_loader, self.val_loader = self.init_dataset(fold=fold)

            # 모델, 옵티마이저, 손실 함수, 학습률 스케줄러 초기화
            self.model, self.optimizer, self.criterion, self.scheduler = self.init_model(
                model_name=self.args.backbone_model,
                learning_rate=self.args.lr,
                outlayer_num=self.args.outlayer_num,
                fold=fold
            )

            # 모델 파라미터 및 FLOPs 계산
            self.calculate_model_params(self.model)

            # WandB 초기화
            if self.wandb_use:
                self.wandb_init()

            # Train
            print(f"\033[41mStart Training - Fold {current_fold} \033[0m")
            val_accuracy, val_loss, best_epoch = self.train_epoch(current_fold)
            fold_scores.append(val_accuracy)

            # 모델이 저장되지 않았다면 마지막 에포크의 모델 저장
            if self.best_model_path is None and best_epoch != -1:
                print("학습 중 개선이 감지되지 않았습니다. 마지막 에포크의 모델을 저장합니다.")
                self.save_best_model(best_epoch, val_loss, current_fold)
            elif self.best_model_path is None:
                print("학습 중 어떤 모델도 저장되지 않았습니다. 현재 모델을 저장합니다.")
                self.save_best_model(epoch=self.args.epochs, val_loss=val_loss, current_fold=current_fold)

            # WandB 세션 종료
            if self.wandb_use:
                wandb.finish()

            # 가비지 컬렉션 실행
            gc.collect()

            # CUDA 캐시 비우기
            torch.cuda.empty_cache()

        # 모든 폴드의 평균 성능 출력
        avg_score = np.mean(fold_scores)
        print(f"\n\033[42m모든 폴드의 평균 검증 정확도: {avg_score:.2f}%\033[0m")

    def train_epoch(self, current_fold: int) -> Tuple[float, float]:
        final_val_accuracy = 0.0
        final_val_loss = float('inf')

        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, masks, labels in self.train_loader:
                self.optimizer.zero_grad()
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).float()
                if self.args.mask_use:
                    inputs = inputs[:, :2, :, :]
                    masks = masks.to(self.device, non_blocking=True)
                    inputs = torch.cat([inputs, masks], dim=1)  # torch.concat -> torch.cat로 변경

                outputs = self.model(inputs)

                if self.args.outlayer_num > 1:
                    # 다중 분류
                    _, predicted = torch.max(outputs.data, 1)
                    labels = labels.long()
                else:
                    # 이진 분류
                    probs = torch.sigmoid(outputs)
                    predicted = (probs > 0.5).float()
                    
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = total_loss / len(self.train_loader)
            epoch_accuracy = 100 * correct / total
            print(f"Epoch [{epoch}/{self.args.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

            # Validation
            print(f"\033[41mStart Validation - Fold {current_fold} \033[0m")
            val_accuracy, val_loss = self.validate(epoch=epoch)
            final_val_accuracy = val_accuracy
            final_val_loss = val_loss

            # 학습률 스케줄러 업데이트
            self.scheduler.step()

            # 조기 종료 로직
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                # 최적 모델 저장
                self.save_best_model(epoch, val_loss, current_fold)
            else:
                self.early_stop_counter += 1
            # 조기 종료 카운터가 patience보다 크거나 같으면 학습 종료
            if self.early_stop_counter >= self.patience:
                print("Early stopping triggered")
                self.save_best_model(epoch, val_loss, current_fold)
                break

        return final_val_accuracy, final_val_loss, epoch   # 반환값 유지

    def validate(self, epoch: int) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0

        all_labels = []
        all_predictions = []
        all_probs = []

        with torch.no_grad():
            for inputs, masks, labels in self.val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).float()
                if self.args.mask_use:
                    inputs = inputs[:, :2, :, :]
                    masks = masks.to(self.device, non_blocking=True)
                    inputs = torch.cat([inputs, masks], dim=1)

                outputs = self.model(inputs)

                if self.args.outlayer_num > 1:
                    # 다중 분류
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                    labels = labels.long()
                else:
                    # 이진 분류
                    probs = torch.sigmoid(outputs)
                    predicted = (probs > 0.5).float()

                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                # 평가 지표 계산을 위해 레이블과 예측값 저장
                all_labels.extend(labels.detach().cpu().numpy())
                all_predictions.extend(predicted.detach().cpu().numpy())
                all_probs.extend(probs.detach().cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)

        # 유효하지 않은 손실 값 처리
        if not np.isfinite(avg_loss):
            print(f"경고: 에포크 {epoch}에서 유효하지 않은 평균 손실이 감지되었습니다. 손실을 무한대로 설정합니다.")
            avg_loss = float('inf')

        if self.args.outlayer_num > 1:
            metrics = multi_classify_metrics(
                y_true=all_labels,
                y_pred=all_predictions,
                y_prob=np.array(all_probs),
                average='weighted'
            )
        else:
            metrics = binary_classify_metrics(
                y_true=all_labels,
                y_pred=all_predictions,
                y_prob=np.array(all_probs),
                test_on=False
            )

        metrics['val_loss'] = avg_loss
        metrics.pop('epoch', None)

        if self.wandb_use:
            wandb.log(metrics, step=epoch)

        print(f"검증 손실: {avg_loss:.4f}, 재현율: {metrics.get('Sensitivity', 0):.2f}%, 정확도: {metrics.get('Accuracy', 0):.2f}%")
        return metrics.get('Accuracy', 0), avg_loss


    def save_best_model(self, epoch: int, val_loss: float, current_fold: int):
        try:
            # 이전 best 모델이 존재한다면 삭제
            if self.best_model_path and os.path.exists(self.best_model_path):
                os.remove(self.best_model_path)
                print(f"이전 최적 모델을 삭제했습니다: {self.best_model_path}")

            # 저장 디렉토리 생성
            save_dir = os.path.join(self.args.save_dir, self.run_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"best_model_fold_{current_fold}_epoch_{epoch}.pth")

            # 모델 상태 저장
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'loss': val_loss,
            }, save_path)
            print(f"최적 모델이 다음 경로에 저장되었습니다: {save_path}")

            # 현재 best 모델 경로 갱신
            self.best_model_path = save_path
        except Exception as e:
            print(f"폴드 {current_fold}의 에포크 {epoch}에서 모델 저장 중 오류 발생: {e}")


class MultiClassifier(BaseClassifier):
    pass  # 추가적인 다중 클래스 전용 기능이 필요하다면 여기에 구현

class BinaryClassifier(BaseClassifier):
    pass  # 추가적인 이진 클래스 전용 기능이 필요하다면 여기에 구현

def main():
    parser = argparse.ArgumentParser(description="PyTorch Multi-Classifier with WandB")
    # Seed
    parser.add_argument("--seed", type=int, default=627, help="Seed for reproducibility")

    # Logging
    parser.add_argument("--wandb_use", type=str, default='no', choices=["yes", "no"], help="WandB 로깅 활성화")
    parser.add_argument('--wandb_project', type=str, default="multi-classifier", help='WandB project name')

    # Data Parameter
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory path')
    parser.add_argument('--csv_path', type=str, required=True, help='CSV file path')
    parser.add_argument('--version', type=str, required=True, help='Model version')
    parser.add_argument('--mask_use', type=str, required=True, help='Mask Using')

    # Model Parameter 
    parser.add_argument('--fold_num', type=int, default=5, help='Number of folds for k-fold cross validation')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Batch size for training dataset')
    parser.add_argument('--valid_batch_size', type=int, default=32, help='Batch size for validation dataset')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--backbone_model', type=str, required=True, help='Backbone Model')
    parser.add_argument('--type', type=str, required=True, help='Model type')
    parser.add_argument('--save_dir', type=str, default="saved_models", help='Directory to save models')
    parser.add_argument('--outlayer_num', type=int, required=True, help='Number of output layers (1 for binary, >1 for multi-class)')
    
    args = parser.parse_args()

    # 분류기 초기화
    if args.outlayer_num == 1:
        classifier = BinaryClassifier(args)
    else:
        classifier = MultiClassifier(args)

if __name__ == '__main__':
    main()

