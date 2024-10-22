#!/usr/bin/env python
import sys
import os
import gc  # 파일 상단에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import argparse
from typing import Tuple, List, Any, Dict

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from ptflops import get_model_complexity_info
from torchsampler import ImbalancedDatasetSampler

import wandb

from lib.seed import set_seed
from lib.dataset import Custom_stratified_Dataset
from lib.datasets.sampler import class_weight_sampler, class_weight_getter
from lib.datasets.ds_tools import kfold_extract
from model.loader import model_Loader
from lib.metric.metrics import multi_classify_metrics, binary_classify_metrics


# 환경 변수 설정 및 백엔드 최적화
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.autograd.set_detect_anomaly(True)

class BaseClassifier:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = self.setup_device()
        self.wandb_use = self.check_wandb_use()
        self.best_accuracy = 0.0
        self.save_every = math.ceil(self.args.epochs / 10)

        # 조기 종료 파라미터
        self.pmuxatience = 50
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0

        print("=" * 200, "\033[41mStart Initialization\033[0m")
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
        # GPU에서만 GradScaler 사용
        if device == 'cuda':
            self.scaler = torch.GradScaler(device)
        else:
            self.scaler = None

        print(f"\033[41mCUDA Status : {device.type}\033[0m")
        return device

    def init_kfold_dataset(self, csv_path, fold_num, seed) -> List[Dict[str, Any]]:
        kfolds, _, _ = kfold_extract(
            csv_path=csv_path,
            n_splits=fold_num,
            random_state=seed,
            shuffle=True
        )
        return kfolds

    def init_augmentation(self) -> Tuple[List[Any], List[Any]]:
        train_augment_list = [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
        valid_augment_list = [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
        return train_augment_list, valid_augment_list

    def init_dataset(self, fold) -> Tuple[DataLoader, DataLoader]:
        train_dataset = Custom_stratified_Dataset(
            df=fold['train'],
            root_dir=self.args.data_dir,
            transform=transforms.Compose(self.args.train_augment_list)
        )
        val_dataset = Custom_stratified_Dataset(
            df=fold['val'],
            root_dir=self.args.data_dir,
            transform=transforms.Compose(self.args.valid_augment_list)
        )
        # sampler = class_weight_sampler(fold)
        sampler = ImbalancedDatasetSampler

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler = sampler(
                train_dataset
            ),
            num_workers=8,
            pin_memory=True if self.device == 'cuda' else False,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.valid_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device == 'cuda' else False,
            drop_last=True
        )

        return train_loader, val_loader

    def init_model(self, model_name: str, learning_rate: float, outlayer_num: int, fold) -> Tuple[nn.Module, optim.Optimizer, nn.Module]:
        model = model_Loader(model_name=model_name, outlayer_num=outlayer_num, type=self.args.type)
        model.to(self.device)

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)  # 가중치 감쇠 추가

        if outlayer_num > 1:
            criterion = nn.CrossEntropyLoss()
        elif outlayer_num == 1:
            pos_weight = class_weight_getter(fold)
            criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight)
        else:
            raise ValueError("outlayer_num must be greater than 0.")

        # 학습률 스케줄러를 ReduceLROnPlateau로 변경
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        
        print("\033[41mFinished Model Initialization\033[0m")
        return model, optimizer, criterion, scheduler


    def calculate_model_params(self, model: nn.Module):
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n\033[0mTotal trainable parameters: {total_params}\033[0m")

        input_res = (3, 224, 224)
        try:
            macs, params = get_model_complexity_info(model, input_res, as_strings=False,
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
        self.kfolds = self.init_kfold_dataset(csv_path = self.args.csv_path, fold_num = self.args.fold_num, seed = self.args.seed)
        fold_scores = []
        set_seed(self.args.seed)

        # 데이터 증강 초기화 (반복문 바깥에서 한 번만 호출)
        self.args.train_augment_list, self.args.valid_augment_list = self.init_augmentation()

        for idx, fold in enumerate(self.kfolds):
            self.run_name = f"{self.args.backbone_model}_{self.args.type}_fold_{idx + 1}-{self.args.version}"
            self.args.fold_num = idx + 1

            # 데이터 로더 초기화
            self.train_loader, self.val_loader = self.init_dataset(fold=fold)

            # 모델, 옵티마이저, 손실 함수, 학습률 스케줄러 초기화
            self.model, self.optimizer, self.criterion, self.scheduler = self.init_model(
                model_name=self.args.backbone_model,
                learning_rate=self.args.lr,
                outlayer_num=self.args.outlayer_num,
                fold = fold
            )

            # 모델 파라미터 및 FLOPs 계산
            self.calculate_model_params(self.model)

            # WandB 초기화
            if self.wandb_use:
                self.wandb_init()

            # Train
            print(f"\033[41mStart Training - Fold {idx + 1} \033[0m")
            val_accuracy, val_loss = self.train_epoch()
            fold_scores.append(val_accuracy)

            # WandB 세션 종료
            if self.wandb_use:
                wandb.finish()

            # # 조기 종료 조건 체크
            # if self.early_stop_counter >= self.patience:
            #     print("조기 종료가 트리거되었습니다.")
            #     break
            # 가비지 컬렉션 실행
            gc.collect()

            # CUDA 캐시 비우기
            torch.cuda.empty_cache()
        # 모든 fold의 평균 성능 출력
        avg_score = np.mean(fold_scores)
        print(f"\n\033[42mAverage Validation Accuracy across all folds: {avg_score:.2f}%\033[0m")

    def train_epoch(self) -> Tuple[float, float]:
        final_val_accuracy = 0.0
        final_val_loss = float('inf')

        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).float()

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                if self.args.outlayer_num > 1:
                    _, predicted = torch.max(outputs.data, 1)
                    labels = labels.long()
                else:
                    probs = torch.sigmoid(outputs)
                    predicted = (probs > 0.4).float()  # 임계값을 0.4로 설정

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = total_loss / len(self.train_loader)
            epoch_accuracy = 100 * correct / total
            print(f"Epoch [{epoch}/{self.args.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

            # Validation
            print(f"\033[41mStart Validation - Fold {self.args.fold_num} \033[0m")
            val_accuracy, val_loss = self.validate(epoch=epoch)
            final_val_accuracy = val_accuracy
            final_val_loss = val_loss

            # 학습률 스케줄러 업데이트
            self.scheduler.step(val_loss)  # 검증 손실을 기반으로 스케줄러 업데이트

            # 조기 종료 로직
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                # 최적 모델 저장
                self.save_best_model(epoch, val_loss)
            else:
                self.early_stop_counter += 1

            if self.early_stop_counter >= self.patience:
                print("Early stopping triggered")
                break

    def validate(self, epoch: int) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        all_labels = []
        all_predictions = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).float()

                # with torch.autocast(device_type = "cuda"):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                if self.args.outlayer_num > 1:
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                    labels = labels.long()
                else:
                    probs = torch.sigmoid(outputs)
                    predicted = (probs > 0.5).float()

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)

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
                y_prob=np.array(all_probs)
            )

        metrics['val_loss'] = avg_loss
        metrics.pop('epoch', None)

        if self.wandb_use:
            wandb.log(metrics, step=epoch)

        print(f"Validation Loss: {avg_loss:.4f}, Recall:{metrics.get('Sensitivity', 0):.2f}%,  Acc: {metrics.get('Accuracy', 0):.2f}%")
        return metrics.get('Accuracy', 0), avg_loss

    def save_best_model(self, epoch: int, val_loss: float):
        save_dir = os.path.join(self.args.save_dir, self.run_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"best_model_fold_{self.args.fold_num}_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': val_loss,
        }, save_path)
        print(f"Best model saved to {save_path}")

    def load_model(self, save_path: str) -> Dict[str, Any]:
        checkpoint = torch.load(save_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from {save_path}, Epoch: {epoch}, Loss: {loss}")
        return checkpoint


class MultiClassifier(BaseClassifier):
    pass  # 추가적인 다중 클래스 전용 기능이 필요하다면 여기에 구현


class BinaryClassifier(BaseClassifier):
    pass  # 추가적인 이진 클래스 전용 기능이 필요하다면 여기에 구현


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Multi-Classifier with WandB")
    # Seed
    parser.add_argument("--seed", type=int, default=627, help="Seed for reproducibility")

    # Logging
    parser.add_argument("--wandb_use", type = str, default = 'no', choices = ["yes", "no"], help="WandB 로깅 활성화")
    parser.add_argument('--wandb_project', type=str, default="multi-classifier", help='WandB project name')

    # Data Parameter
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory path')
    parser.add_argument('--csv_path', type=str, required=True, help='CSV file path')
    parser.add_argument('--version', type=str, required=True, help='model version')

    # Model Parameter 
    parser.add_argument('--fold_num', type=int, default=5, help='Number of folds for k-fold cross validation')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Batch size for training dataset')  # 적절한 배치 사이즈로 수정
    parser.add_argument('--valid_batch_size', type=int, default=32, help='Batch size for validation dataset')  # 적절한 배치 사이즈로 수정
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')  # 수정
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')  # 200으로 설정
    parser.add_argument('--backbone_model', type=str, required=True, help='Backbone Model')
    parser.add_argument('--type', type=str, required=True, help='Model type')
    parser.add_argument('--save_dir', type=str, default="saved_models", help='Directory to save models')  # 추가
    parser.add_argument('--outlayer_num', type=int, help='Number of output layers (1 for binary, >1 for multi-class)')  # 추가
    
    args = parser.parse_args()

    # 분류기 초기화
    if args.outlayer_num == 1:
        classifier = BinaryClassifier(args)
    else:
        classifier = MultiClassifier(args)


if __name__ == '__main__':
    main()
