#%% Default Libs
from dotenv import load_dotenv
from lib.seed import seed_prefix 
import sys, os 
seed_prefix(seed=42)
load_dotenv('../.env')

import datetime
import json

#%% Exp Lib
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, auc, roc_curve
import numpy as np
import wandb
import logging
#%% [EXP] Custom Lib
from data import workflow_data, workflow_k_fold_data
from models import Model_Loader
from lib.type import convert_np_number_to_python
class PCOS_trainer:
    def __init__(self, args):
        self.args = args
        self.exp_date = args.exp_date
            
    def k_fold_fit(self):
        fold_auc_list = []
        for fold_num in range(self.args.k_fold): # K-Fold 반복
            self.args.fold_num = fold_num # fold_num 설정
            
            # Logging 및 wandb 초기화 (fold마다 별도 로그 파일 생성)
            logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                                filename=f"log/{self.exp_date}_fold{self.args.fold_num + 1}.log",
                                level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S')
            logging.info(f"Fold {self.args.fold_num + 1} 시작")
            if getattr(self.args, "use_wandb", False):
                wandb.init(project=self.args.project_name,
                        name=f"{self.args.experiment_name}_fold{self.args.fold_num + 1}",
                        config=vars(self.args))
            
            # 각 fold에 대해 데이터, 모델, 실험 세팅 초기화
            self.init_data()
            self.init_model()
            self.init_exp_setting()
            
            for epoch in range(1, self.args.epoch+1):
                if epoch <= self.warmup_epochs:
                    lr_scale = epoch / self.warmup_epochs
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = self.args.lr * lr_scale
                    current_lr = self.args.lr * lr_scale
                    logging.info(f"Warmup Epoch {epoch}: setting lr to {current_lr:.6f}")
                else:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                
                train_loss, train_metrics = self.trainer()
                val_loss, val_metrics = self.validation()
                
                if epoch > self.warmup_epochs:
                    self.scheduler.step()
                    # self.scheduler.step(val_metrics['auc'])
                
                if getattr(self.args, "use_wandb", False):
                    log_dict = {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_auc": train_metrics['auc'],
                        "train_f1": train_metrics['f1'],
                        "train_accuracy": train_metrics['accuracy'],
                        "train_recall": train_metrics['recall'],
                        "val_loss": val_loss,
                        "val_auc": val_metrics['auc'],
                        "val_f1": val_metrics['f1'],
                        "val_accuracy": val_metrics['accuracy'],
                        "val_recall": val_metrics['recall'],
                        "lr": current_lr
                    }
                    train_recall_per_class = train_metrics.get('recall_per_class', [])
                    for i, rec_val in enumerate(train_recall_per_class):
                        log_dict[f"train_recall_class_{i}"] = rec_val

                    train_f1_per_class = train_metrics.get('f1_per_class', [])
                    for i, f1_val in enumerate(train_f1_per_class):
                        log_dict[f"train_f1_class_{i}"] = f1_val

                    train_accuracy_per_class = train_metrics.get('accuracy_per_class', [])
                    for i, acc_val in enumerate(train_accuracy_per_class):
                        log_dict[f"train_accuracy_class_{i}"] = acc_val

                    val_recall_per_class = val_metrics.get('recall_per_class', [])
                    for i, rec_val in enumerate(val_recall_per_class):
                        log_dict[f"val_recall_class_{i}"] = rec_val

                    val_f1_per_class = val_metrics.get('f1_per_class', [])
                    for i, f1_val in enumerate(val_f1_per_class):
                        log_dict[f"val_f1_class_{i}"] = f1_val

                    val_accuracy_per_class = val_metrics.get('accuracy_per_class', [])
                    for i, acc_val in enumerate(val_accuracy_per_class):
                        log_dict[f"val_accuracy_class_{i}"] = acc_val
                    
                    wandb.log(log_dict)
                
                logging.info(f"[Epoch {epoch}/{self.args.epoch}] "
                            f"Train Loss: {train_loss:.4f}, Train AUC: {train_metrics['auc']:.4f}, "
                            f"Train F1: {train_metrics['f1']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}, Train Recall: {train_metrics['recall']:.4f} | "
                            f"Val Loss: {val_loss:.4f}, Val AUC: {val_metrics['auc']:.4f}, "
                            f"Val F1: {val_metrics['f1']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val Recall: {val_metrics['recall']:.4f}")

                if val_metrics['auc'] > self.best_val_auc:
                    self.best_val_auc = val_metrics['auc']
                    self.no_improve_count = 0

                    # 각 fold별로 모델 저장 경로를 분리하여 저장
                    save_model_path = f"/home/eiden/eiden/PCOS-roi-classification/v2/log/{self.exp_date}_fold{self.args.fold_num + 1}.pth"
                    torch.save(self.model.state_dict(), save_model_path)

                    save_json_path = f"/home/eiden/eiden/PCOS-roi-classification/v2/log/{self.exp_date}_fold{self.args.fold_num + 1}.json"
                    with open(save_json_path, 'w') as f:
                        json.dump({
                            **vars(self.args),
                            **{
                                'best_val_auc': self.best_val_auc,
                                'train_metrics': convert_np_number_to_python(train_metrics),
                                'val_metrics': convert_np_number_to_python(val_metrics),
                                'exp_date': self.exp_date,
                                'model_path': save_model_path
                            }
                        }, f)
                    
                    logging.info(f"  >> Best model updated! (AUC={self.best_val_auc:.4f}) <<")
                else:
                    self.no_improve_count += 1
                
                if self.no_improve_count >= self.args.patience:
                    logging.info("Early stopping triggered!")
                    break
            
            if getattr(self.args, "use_wandb", False):
                wandb.finish()
            
            # 각 fold의 결과 저장
            fold_auc_list.append(self.best_val_auc)
            logging.info(f"Fold {self.args.fold_num + 1} 완료 - Best AUC: {self.best_val_auc:.4f}")
        
        # 모든 fold의 평균 AUC 계산 후 반환
        avg_auc = sum(fold_auc_list) / len(fold_auc_list)
        logging.info(f"전체 Fold 평균 AUC: {avg_auc:.4f}")
        return avg_auc

    def fit(self):
        """전체 학습 루프: Training + Validation + EarlyStopping/ModelSave"""
            #%% Logging 설정
        logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', filename = f"log/{self.exp_date}.log", level = logging.INFO, datefmt='%m/%d/%Y %I:%M:%S')
        if getattr(self.args, "use_wandb", False):
            wandb.init(project=self.args.project_name, name=self.args.experiment_name, config=vars(self.args))

        self.init_data()
        self.init_model()
        self.init_exp_setting()
        
        for epoch in range(1, self.args.epoch+1):
            # Warmup: 초기 warmup_epochs 동안 선형적으로 lr 증가
            if epoch <= self.warmup_epochs:
                lr_scale = epoch / self.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.args.lr * lr_scale
                current_lr = self.args.lr * lr_scale
                logging.info(f"Warmup Epoch {epoch}: setting lr to {current_lr:.6f}")
            else:
                current_lr = self.optimizer.param_groups[0]["lr"]
            
            train_loss, train_metrics = self.trainer()  # [1] Training
            val_loss, val_metrics = self.validation()     # [2] Validation

            # warmup이 끝난 이후부터 scheduler 적용
            if epoch > self.warmup_epochs:
                self.scheduler.step()
                # self.scheduler.step(val_metrics['auc'])
            
            # wandb에 로그 기록
            if getattr(self.args, "use_wandb", False):
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_auc": train_metrics['auc'],
                    "train_f1": train_metrics['f1'],
                    "train_accuracy": train_metrics['accuracy'],
                    "train_recall": train_metrics['recall'],
                    "train_auc_per_class": train_metrics.get('auc_per_class', []),
                    "train_f1_per_class": train_metrics.get('f1_per_class', []),
                    "train_recall_per_class": train_metrics.get('recall_per_class', []),
                    "train_accuracy_per_class": train_metrics.get('accuracy_per_class', []),
                    "val_loss": val_loss,
                    "val_auc": val_metrics['auc'],
                    "val_f1": val_metrics['f1'],
                    "val_accuracy": val_metrics['accuracy'],
                    "val_recall": val_metrics['recall'],
                    "val_auc_per_class": val_metrics.get('auc_per_class', []),
                    "val_f1_per_class": val_metrics.get('f1_per_class', []),
                    "val_recall_per_class": val_metrics.get('recall_per_class', []),
                    "val_accuracy_per_class": val_metrics.get('accuracy_per_class', []),
                    "lr": current_lr
                })
            
            logging.info(f"[Epoch {epoch}/{self.args.epoch}] "
                f"Train Loss: {train_loss:.4f}, Train AUC: {train_metrics['auc']:.4f}, "
                f"Train F1: {train_metrics['f1']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}, Train Recall: {train_metrics['recall']:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val AUC: {val_metrics['auc']:.4f}, "
                f"Val F1: {val_metrics['f1']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val Recall: {val_metrics['recall']:.4f}")
            
            # Early Stopping + Model Save
            if val_metrics['auc'] > self.best_val_auc:
                self.best_val_auc = val_metrics['auc']
                self.no_improve_count = 0

                save_model_path = f"/home/eiden/eiden/PCOS-roi-classification/v2/log/{self.exp_date}.pth"
                torch.save(self.model.state_dict(), save_model_path)

                save_json_path = f"/home/eiden/eiden/PCOS-roi-classification/v2/log/{self.exp_date}.json"
                with open(save_json_path, 'w') as f:
                    json.dump({
                        'best_val_auc': self.best_val_auc,
                        'train_metrics': train_metrics,
                        'val_metrics': val_metrics,
                        'exp_date': self.exp_date,
                        'model_path': save_model_path
                    }, f)
                
                logging.info(f"  >> Best model updated! (AUC={self.best_val_auc:.4f}) <<")
            else:
                self.no_improve_count += 1
            
            if self.no_improve_count >= self.args.patience:
                logging.info("Early stopping triggered!")
                break
        
        if getattr(self.args, "use_wandb", False):
            wandb.finish()
        
        return self.best_val_auc

    def init_data(self):
        """데이터 로더 초기화"""
        # use_kfold 플래그가 True이면 미리 분리한 데이터를 사용
        if getattr(self.args, "use_kfold", False):
            self.train_loader, self.valid_loader = workflow_k_fold_data(
                datasheet_path=self.args.datasheet_path,
                data_dir=self.args.data_dir,
                sample_data_dir= self.args.sample_data_dir,
                sample_datasheet_path= self.args.sample_datasheet_path,
                sampler=self.args.sampler_name,
                n_splits=self.args.k_fold,
                fdx = self.args.fold_num,
                binary_use = self.args.binary_use,
                borderline_use = self.args.borderline_use,
            )
        else:
            from data import workflow_data  # 기존 분할 함수
            self.train_loader, self.valid_loader = workflow_data(
                datasheet_path=self.args.datasheet_path,
                data_dir=self.args.data_dir,
                sample_data_dir= self.args.sample_data_dir,
                sample_datasheet_path= self.args.sample_datasheet_path,
                sampler=self.args.sampler_name,
                binary_use = self.args.binary_use,
                borderline_use = self.args.borderline_use,
            )
    
    def init_model(self):
        """모델 초기화"""
        self.num_classes = 1 if self.args.binary_use else 3
        # 예시: torchvision의 resnet18를 사용하여 분류 모델 구축 (3 클래스 분류)
        self.model = Model_Loader(model_name = self.args.model_name + '_' + self.args.model_version, num_classes = self.num_classes).to(self.args.device)
        
    def init_exp_setting(self):
        """실험 세팅 (Loss, Optim, Scheduler, 모니터링 변수 등)"""
        # 이진 분류일 경우 BCEWithLogitsLoss, 다중 클래스일 경우 CrossEntropyLoss 사용
        if self.args.binary_use:
            self.criterion = nn.BCEWithLogitsLoss().to(self.args.device)
        else:
            if self.args.loss_name == 'polyl1ce':
                from loss import Poly1CrossEntropyLoss
                self.criterion = Poly1CrossEntropyLoss(num_classes = self.num_classes, reduction = 'mean').to(self.args.device)
            elif self.args.loss_name == 'poly1focal':
                from loss import Poly1FocalLoss
                if self.args.binary_use:
                    self.criterion = Poly1FocalLoss(
                        num_classes = self.num_classes,
                        reduction = 'mean',
                        label_is_onehot = True,
                        pos_weight = torch.tensor([3.]).to(self.args.device)
                    ).to(self.args.device)
                else:
                    self.criterion = Poly1FocalLoss(
                        num_classes = self.num_classes, 
                        reduction = 'mean', 
                        label_is_onehot = False, 
                        pos_weight = torch.tensor([1., 5., 5.]).reshape([1, self.num_classes]).to(self.args.device)
                    ).to(self.args.device)
            else:
                self.criterion = nn.CrossEntropyLoss(label_smoothing=self.args.loss_label_smoothing).to(self.args.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.warmup_epochs = self.args.warmup_epochs

        #%% [Scheduler] Series 
        # 스케줄러 타입에 따라 학습률 조정 방식을 통일된 파라미터로 설정합니다.
        if self.args.scheduler_type == "plateau":
            # ReduceLROnPlateau: 검증 지표(여기서는 AUC)가 개선되지 않으면 학습률을 감소시킵니다.
            # mode='max'로 설정하여 AUC가 최대화될 때까지 기다리고, factor와 patience는 감소 비율과 대기 에폭 수를 지정합니다.
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=self.args.scheduler_factor, patience=self.args.scheduler_patience, verbose=True)
        elif self.args.scheduler_type == "ExponentialLR":
            # ExponentialLR: 매 에폭마다 학습률을 gamma 배 만큼 감소시킵니다.
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.args.scheduler_gamma)
        elif self.args.scheduler_type == "cosine":
            # CosineAnnealingLR: 에폭 수에 따라 코사인 형태로 학습률을 점진적으로 감소시킵니다.
            # T_max는 주기(최대 에폭 수)이며, eta_min은 최저 학습률입니다.
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.scheduler_T_max, eta_min=self.args.scheduler_eta_min)
        elif self.args.scheduler_type == "StepLR":
            # StepLR: 지정한 step_size마다 학습률을 gamma 배 만큼 감소시킵니다.
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.scheduler_step_size, gamma=self.args.scheduler_gamma)
        elif self.args.scheduler_type == "MultiStepLR":
            # MultiStepLR: milestones에 지정한 에폭마다 학습률을 gamma 배 만큼 감소시킵니다.
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.scheduler_milestones, gamma=self.args.scheduler_gamma)
        elif self.args.scheduler_type == "CyclicLR":
            # CyclicLR: 학습률을 base_lr와 max_lr 사이에서 주기적으로 변화시키는 방식입니다.
            # step_size_up과 step_size_down은 학습률이 상승/하강하는 단계의 에폭 수를 의미하며,
            # mode='triangular2'는 주기가 지날 때마다 최대 학습률을 반으로 줄이는 설정입니다.
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.args.lr, max_lr=self.args.lr * self.args.scheduler_max_lr_scale, step_size_up=self.args.scheduler_step_size_up, step_size_down=self.args.scheduler_step_size_down, gamma=self.args.scheduler_gamma, mode='triangular2')
        elif self.args.scheduler_type == "CosineAnnealingWarmRestarts":
            # CosineAnnealingWarmRestarts: 코사인 학습률 스케줄러를 사용하되, 주기마다 학습률을 재시작(restart)합니다.
            # T_0는 첫 주기의 에폭 수, T_mult는 이후 주기의 배수, eta_min은 최소 학습률을 의미합니다.
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=self.args.scheduler_T0, T_mult=self.args.scheduler_T_mult, eta_min=self.args.lr * self.args.scheduler_eta_min_scale)
        else:
            # 정의되지 않은 스케줄러 타입이 전달된 경우 에러 발생
            raise ValueError(f"Invalid Scheduler Type: {self.args.scheduler_type}")

        self.best_val_auc = 0.0
        self.no_improve_count = 0

    def calculate_metrics(self, labels, preds):
        """
        labels: numpy array or torch tensor (정답 라벨)
        preds: numpy array (예측 확률, shape: [N, num_classes])
        """
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy().flatten()
        if torch.is_tensor(preds):
            if self.args.binary_use:
                preds = preds.cpu().numpy().flatten()
            else:
                preds = preds.cpu().numpy()
        
        metrics = {}
        if self.args.binary_use:
            try:
                metrics['auc'] = roc_auc_score(labels, preds)
            except Exception as e:
                metrics['auc'] = 0.0
            metrics['best_thr'] = self.calculate_best_thr(labels, preds)
            pred_class = (preds >= metrics['best_thr']).astype(int)
            metrics['f1'] = f1_score(labels, pred_class)
            metrics['accuracy'] = accuracy_score(labels, pred_class)
            metrics['recall'] = recall_score(labels, pred_class)
        else:
            try:
                metrics['auc'] = roc_auc_score(labels, preds, multi_class='ovr', average='macro')
            except Exception as e:
                metrics['auc'] = 0.0
            preds_class = preds.argmax(axis=1)
            metrics['f1'] = f1_score(labels, preds_class, average='macro')
            metrics['accuracy'] = accuracy_score(labels, preds_class)
            metrics['recall'] = recall_score(labels, preds_class, average='macro')
            
            num_classes = preds.shape[1]
            auc_per_class = []
            for i in range(num_classes):
                y_true_i = (labels == i).astype(int)
                y_score_i = preds[:, i]
                try:
                    auc_i = roc_auc_score(y_true_i, y_score_i)
                except Exception as e:
                    auc_i = 0.0
                auc_per_class.append(auc_i)
            metrics['auc_per_class'] = auc_per_class

            f1_per_class = f1_score(labels, preds_class, average=None)
            recall_per_class = recall_score(labels, preds_class, average=None)
            metrics['f1_per_class'] = f1_per_class.tolist()
            metrics['recall_per_class'] = recall_per_class.tolist()

            accuracy_per_class = []
            for i in range(num_classes):
                mask = (labels == i)
                if mask.sum() == 0:
                    acc_i = 0.0
                else:
                    correct = (preds_class[mask] == i).sum()
                    acc_i = correct / mask.sum()
                accuracy_per_class.append(acc_i)
            metrics['accuracy_per_class'] = accuracy_per_class

        return metrics

    def trainer(self):
        """하나의 Epoch에 대한 Training Loop"""
        self.model.train()
        train_loss = 0.0
        all_preds = []
        all_labels = []
        
        for X, y in self.train_loader:
            X = X.to(self.args.device)
            y = y.to(self.args.device)
            
            self.optimizer.zero_grad()
            y_res = self.model(X)  # [B x num_classes] 또는 [B x 1]
            
            # 이진 분류일 경우 Float 타입으로 변환
            if self.args.binary_use:
                y = y.float()
                
            loss = self.criterion(y_res, y)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            if self.args.binary_use:
                prob = torch.sigmoid(y_res).detach().cpu()
            else:
                prob = F.softmax(y_res, dim=1).detach().cpu()
            all_preds.append(prob)
            all_labels.append(y.detach().cpu())
        
        train_loss = train_loss / len(self.train_loader)
        train_preds = torch.cat(all_preds, dim=0)
        train_labels = torch.cat(all_labels, dim=0)
        train_metrics = self.calculate_metrics(train_labels, train_preds)
        return train_loss, train_metrics
    
    def validation(self):
        """하나의 Epoch에 대한 Validation Loop"""
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_val, y_val in self.valid_loader:
                X_val = X_val.to(self.args.device)
                y_val = y_val.to(self.args.device)
                
                # 이진 분류일 경우 Float 타입으로 변환
                if self.args.binary_use:
                    y_val = y_val.float()
                    
                y_res_val = self.model(X_val) # [B x num_classes]
                loss_val = self.criterion(y_res_val, y_val)
                val_loss += loss_val.item()
                
                if self.args.binary_use:
                    prob = torch.sigmoid(y_res_val).cpu()
                else:
                    prob = F.softmax(y_res_val, dim=1).cpu()
                all_preds.append(prob)
                all_labels.append(y_val.cpu())
        
        val_loss = val_loss / len(self.valid_loader)
        val_preds = torch.cat(all_preds, dim=0)
        val_labels = torch.cat(all_labels, dim=0)
        
        val_metrics = self.calculate_metrics(val_labels, val_preds)
        return val_loss, val_metrics
    
    def calculate_best_thr(self, labels, preds):
        fpr, tpr, thresholds = roc_curve(labels, preds)
        # Youden's J statistic: tpr - fpr 최대값을 기준으로 optimal threshold 선택
        J = tpr - fpr
        idx_opt = np.argmax(J)
        return thresholds[idx_opt]
        
# ======================= #
# Main 실행부
# ======================= #
def exp1():
    class Args:
        pass
    args = Args()
    
    #%% [WanDB]
    args.use_wandb = True
    args.project_name = "PCOS_Classification"

    #%% [Logging]
    args.exp_date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    #%% [Setting]
    args.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    
    #%% [Data]
    args.datasheet_path = os.getenv('DATASHEET_PATH')
    args.data_dir = os.getenv('DATA_DIR')
    args.borderline_use = True  # Borderline 사용시 : True / Borderline 사용안할시 : False
    
    #%% [Sample Data] : 데이터 분할 후 테스트 진행 시 사용
    args.sample_data_use = False 
    args.sample_datasheet_path = os.getenv("SAMPLE_DATASHEET_PATH") if args.sample_data_use else None
    args.sample_data_dir = os.getenv("SAMPLE_DATA_DIR") if args.sample_data_use else None
    
    args.binary_use = True # 보더라인을 양성에 붙힌 경우 : 0.7246253000224796 / 보더라인을 음성에 붙힌 경우 : 0.7071438091142334 / Multi Class : 0.7119426832054033
    # args.sampler_name = 'balanced' # Default : None / 'balanced' / 'weighted' -> 0.6457(사용X) / 0.5994(사용시)
    args.sampler_name = None # Default : None / 'balanced' / 'weighted' -> 0.6457(사용X) / 0.5994(사용시)
    args.use_kfold = True
    args.k_fold = 5
    
    #%% [Model]
    args.model_name = 'dinov2'
    args.model_version = 'vits14_lc'
    
    #%% [Hyperparameters]
    args.loss_name = 'polyl1ce' # None / 'polyl1ce' / 'poly1focal'
    args.epoch = 50
    args.warmup_epochs = 20
    args.patience = 18 # Best 0.6399 : 최적
    args.loss_label_smoothing = 0.028280726462559483 # 최적 0.864
    # args.lr = 0.0000063301 # 최적 0.864
    args.lr = 0.0000013301 # 최적 

    #%% [Scheduler]
    args.scheduler_type = 'cosine' # 'plateau' / 'cosine' / 'StepLR' / 'MultiStepLR' / 'CyclicLR' / 'CosineAnnealingWarmRestarts'
    args.scheduler_factor = 0.1  # ReduceLROnPlateau에서 사용: 검증 지표가 개선되지 않을 때 학습률을 몇 배로 줄일지 결정 (예: 0.1은 10배 감소)
    args.scheduler_patience = args.patience // 2  # ReduceLROnPlateau에서 사용: 몇 에폭 동안 개선이 없을 때 학습률을 감소시킬지 결정하는 patience 값
    args.scheduler_T_max = args.epoch - args.warmup_epochs  # CosineAnnealingLR에서 사용: 코사인 주기의 길이를 지정 (warmup 이후 전체 에폭 수)
    args.scheduler_eta_min = 1e-6  # CosineAnnealingLR에서 사용: 학습률의 최저 한계를 지정
    args.scheduler_step_size = 10  # StepLR에서 사용: 몇 에폭마다 학습률을 감소시킬지 결정하는 step_size
    args.scheduler_gamma = 0.5  # StepLR, MultiStepLR, CyclicLR 등에서 사용: 학습률 감소 시 적용할 곱셈 계수 (예: 0.5는 50% 감소)
    args.scheduler_milestones = [10, 30, 50]  # MultiStepLR에서 사용: 학습률을 감소시킬 에폭의 리스트
    args.scheduler_max_lr_scale = 10.0  # CyclicLR에서 사용: 최대 학습률을 결정할 때 기본 lr에 곱해질 배수 (예: 10배)
    args.scheduler_step_size_up = 5  # CyclicLR에서 사용: 학습률이 상승하는 단계(에폭 수)를 지정
    args.scheduler_step_size_down = 5  # CyclicLR에서 사용: 학습률이 하강하는 단계(에폭 수)를 지정
    args.scheduler_T0 = 10  # CosineAnnealingWarmRestarts에서 사용: 첫 주기의 길이를 지정하는 파라미터
    args.scheduler_T_mult = 1  # CosineAnnealingWarmRestarts에서 사용: 이후 주기의 주기 배수를 결정 (1이면 주기가 동일)
    args.scheduler_eta_min_scale = 0.1  # CosineAnnealingWarmRestarts에서 사용: 최소 학습률을 결정할 때 기본 lr에 곱해질 배수 (예: 0.1은 10% 수준)

    #%% [Experiment Name]
    # args.experiment_name = f"{args.model_name}_{args.model_version}_Exp_borderline2Malignant"
    args.experiment_name = f"{args.model_name}_{args.model_version}_Exp_[Experiment]polyl1ce"
    
    # Trainer 생성 및 학습 실행
    trainer = PCOS_trainer(args)
    if args.use_kfold:
        avg_auc = trainer.k_fold_fit()
        # log 파일로 평균 AUC 저장
        with open(f"log/{args.exp_date}_avg_auc.txt", 'w') as f:
            f.write(str(avg_auc))
    else:
        avg_auc = trainer.fit()
        with open(f"log/{args.exp_date}_avg_auc.txt", 'w') as f:
            f.write(str(avg_auc))
            
# ======================= #
def exp2():
    class Args:
        pass
    args = Args()
    
    #%% [WanDB]
    args.use_wandb = True
    args.project_name = "PCOS_Classification"

    #%% [Logging]
    args.exp_date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    #%% [Setting]
    args.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    
    #%% [Data]
    args.datasheet_path = os.getenv('DATASHEET_PATH')
    args.data_dir = os.getenv('DATA_DIR')
    args.borderline_use = True  # Borderline 사용시 : True / Borderline 사용안할시 : False
    
    #%% [Sample Data] : 데이터 분할 후 테스트 진행 시 사용
    args.sample_data_use = False 
    args.sample_datasheet_path = os.getenv("SAMPLE_DATASHEET_PATH") if args.sample_data_use else None
    args.sample_data_dir = os.getenv("SAMPLE_DATA_DIR") if args.sample_data_use else None
    
    args.binary_use = True # 보더라인을 양성에 붙힌 경우 : 0.7246253000224796 / 보더라인을 음성에 붙힌 경우 : 0.7071438091142334 / Multi Class : 0.7119426832054033
    # args.sampler_name = 'balanced' # Default : None / 'balanced' / 'weighted' -> 0.6457(사용X) / 0.5994(사용시)
    args.sampler_name = None # Default : None / 'balanced' / 'weighted' -> 0.6457(사용X) / 0.5994(사용시)
    args.use_kfold = True
    args.k_fold = 5
    
    #%% [Model]
    args.model_name = 'dinov2'
    args.model_version = 'vits14_lc'
    
    #%% [Hyperparameters]
    args.loss_name = 'poly1focal' # None / 'polyl1ce' / 'poly1focal'
    args.epoch = 50
    args.warmup_epochs = 20
    args.patience = 18 # Best 0.6399 : 최적
    args.loss_label_smoothing = 0.028280726462559483 # 최적 0.864
    # args.lr = 0.0000063301 # 최적 0.864
    args.lr = 0.0000013301 # 최적 

    #%% [Scheduler]
    args.scheduler_type = 'cosine' # 'plateau' / 'cosine' / 'StepLR' / 'MultiStepLR' / 'CyclicLR' / 'CosineAnnealingWarmRestarts'
    args.scheduler_factor = 0.1  # ReduceLROnPlateau에서 사용: 검증 지표가 개선되지 않을 때 학습률을 몇 배로 줄일지 결정 (예: 0.1은 10배 감소)
    args.scheduler_patience = args.patience // 2  # ReduceLROnPlateau에서 사용: 몇 에폭 동안 개선이 없을 때 학습률을 감소시킬지 결정하는 patience 값
    args.scheduler_T_max = args.epoch - args.warmup_epochs  # CosineAnnealingLR에서 사용: 코사인 주기의 길이를 지정 (warmup 이후 전체 에폭 수)
    args.scheduler_eta_min = 1e-6  # CosineAnnealingLR에서 사용: 학습률의 최저 한계를 지정
    args.scheduler_step_size = 10  # StepLR에서 사용: 몇 에폭마다 학습률을 감소시킬지 결정하는 step_size
    args.scheduler_gamma = 0.5  # StepLR, MultiStepLR, CyclicLR 등에서 사용: 학습률 감소 시 적용할 곱셈 계수 (예: 0.5는 50% 감소)
    args.scheduler_milestones = [10, 30, 50]  # MultiStepLR에서 사용: 학습률을 감소시킬 에폭의 리스트
    args.scheduler_max_lr_scale = 10.0  # CyclicLR에서 사용: 최대 학습률을 결정할 때 기본 lr에 곱해질 배수 (예: 10배)
    args.scheduler_step_size_up = 5  # CyclicLR에서 사용: 학습률이 상승하는 단계(에폭 수)를 지정
    args.scheduler_step_size_down = 5  # CyclicLR에서 사용: 학습률이 하강하는 단계(에폭 수)를 지정
    args.scheduler_T0 = 10  # CosineAnnealingWarmRestarts에서 사용: 첫 주기의 길이를 지정하는 파라미터
    args.scheduler_T_mult = 1  # CosineAnnealingWarmRestarts에서 사용: 이후 주기의 주기 배수를 결정 (1이면 주기가 동일)
    args.scheduler_eta_min_scale = 0.1  # CosineAnnealingWarmRestarts에서 사용: 최소 학습률을 결정할 때 기본 lr에 곱해질 배수 (예: 0.1은 10% 수준)

    #%% [Experiment Name]
    # args.experiment_name = f"{args.model_name}_{args.model_version}_Exp_borderline2Malignant"
    args.experiment_name = f"{args.model_name}_{args.model_version}_Exp_[Experiment]poly1focal"
    
    # Trainer 생성 및 학습 실행
    trainer = PCOS_trainer(args)
    if args.use_kfold:
        avg_auc = trainer.k_fold_fit()
        # log 파일로 평균 AUC 저장
        with open(f"log/{args.exp_date}_avg_auc.txt", 'w') as f:
            f.write(str(avg_auc))
    else:
        avg_auc = trainer.fit()
        with open(f"log/{args.exp_date}_avg_auc.txt", 'w') as f:
            f.write(str(avg_auc))

# ======================= #
def exp3():
    class Args:
        pass
    args = Args()
    
    #%% [WanDB]
    args.use_wandb = True
    args.project_name = "PCOS_Classification"

    #%% [Logging]
    args.exp_date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    #%% [Setting]
    args.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    
    #%% [Data]
    args.datasheet_path = os.getenv('DATASHEET_PATH')
    args.data_dir = os.getenv('DATA_DIR')
    args.borderline_use = True  # Borderline 사용시 : True / Borderline 사용안할시 : False
    
    #%% [Sample Data] : 데이터 분할 후 테스트 진행 시 사용
    args.sample_data_use = False 
    args.sample_datasheet_path = os.getenv("SAMPLE_DATASHEET_PATH") if args.sample_data_use else None
    args.sample_data_dir = os.getenv("SAMPLE_DATA_DIR") if args.sample_data_use else None
    
    args.binary_use = True # 보더라인을 양성에 붙힌 경우 : 0.7246253000224796 / 보더라인을 음성에 붙힌 경우 : 0.7071438091142334 / Multi Class : 0.7119426832054033
    # args.sampler_name = 'balanced' # Default : None / 'balanced' / 'weighted' -> 0.6457(사용X) / 0.5994(사용시)
    args.sampler_name = None # Default : None / 'balanced' / 'weighted' -> 0.6457(사용X) / 0.5994(사용시)
    args.use_kfold = True
    args.k_fold = 5
    
    #%% [Model]
    args.model_name = 'dinov2'
    args.model_version = 'vits14_lc'
    
    #%% [Hyperparameters]
    args.loss_name = 'CE' # None / 'polyl1ce' / 'poly1focal'
    args.epoch = 50
    args.warmup_epochs = 20
    args.patience = 18 # Best 0.6399 : 최적
    args.loss_label_smoothing = 0.028280726462559483 # 최적 0.864
    # args.lr = 0.0000063301 # 최적 0.864
    args.lr = 0.0000013301 # 최적 

    #%% [Scheduler]
    args.scheduler_type = 'cosine' # 'plateau' / 'cosine' / 'StepLR' / 'MultiStepLR' / 'CyclicLR' / 'CosineAnnealingWarmRestarts'
    args.scheduler_factor = 0.1  # ReduceLROnPlateau에서 사용: 검증 지표가 개선되지 않을 때 학습률을 몇 배로 줄일지 결정 (예: 0.1은 10배 감소)
    args.scheduler_patience = args.patience // 2  # ReduceLROnPlateau에서 사용: 몇 에폭 동안 개선이 없을 때 학습률을 감소시킬지 결정하는 patience 값
    args.scheduler_T_max = args.epoch - args.warmup_epochs  # CosineAnnealingLR에서 사용: 코사인 주기의 길이를 지정 (warmup 이후 전체 에폭 수)
    args.scheduler_eta_min = 1e-6  # CosineAnnealingLR에서 사용: 학습률의 최저 한계를 지정
    args.scheduler_step_size = 10  # StepLR에서 사용: 몇 에폭마다 학습률을 감소시킬지 결정하는 step_size
    args.scheduler_gamma = 0.5  # StepLR, MultiStepLR, CyclicLR 등에서 사용: 학습률 감소 시 적용할 곱셈 계수 (예: 0.5는 50% 감소)
    args.scheduler_milestones = [10, 30, 50]  # MultiStepLR에서 사용: 학습률을 감소시킬 에폭의 리스트
    args.scheduler_max_lr_scale = 10.0  # CyclicLR에서 사용: 최대 학습률을 결정할 때 기본 lr에 곱해질 배수 (예: 10배)
    args.scheduler_step_size_up = 5  # CyclicLR에서 사용: 학습률이 상승하는 단계(에폭 수)를 지정
    args.scheduler_step_size_down = 5  # CyclicLR에서 사용: 학습률이 하강하는 단계(에폭 수)를 지정
    args.scheduler_T0 = 10  # CosineAnnealingWarmRestarts에서 사용: 첫 주기의 길이를 지정하는 파라미터
    args.scheduler_T_mult = 1  # CosineAnnealingWarmRestarts에서 사용: 이후 주기의 주기 배수를 결정 (1이면 주기가 동일)
    args.scheduler_eta_min_scale = 0.1  # CosineAnnealingWarmRestarts에서 사용: 최소 학습률을 결정할 때 기본 lr에 곱해질 배수 (예: 0.1은 10% 수준)

    #%% [Experiment Name]
    # args.experiment_name = f"{args.model_name}_{args.model_version}_Exp_borderline2Malignant"
    args.experiment_name = f"{args.model_name}_{args.model_version}_Exp_[Experiment]CE"
    
    # Trainer 생성 및 학습 실행
    trainer = PCOS_trainer(args)
    if args.use_kfold:
        avg_auc = trainer.k_fold_fit()
        # log 파일로 평균 AUC 저장
        with open(f"log/{args.exp_date}_avg_auc.txt", 'w') as f:
            f.write(str(avg_auc))
    else:
        avg_auc = trainer.fit()
        with open(f"log/{args.exp_date}_avg_auc.txt", 'w') as f:
            f.write(str(avg_auc))
            
            
            
if __name__ == "__main__":
    exp1()
    exp2()
    exp3()