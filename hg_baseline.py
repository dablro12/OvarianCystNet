# hg_tuning.py
import sys 
sys.path.append("/workspace/notebooks")

import os
import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)
import evaluate

from utils.dataset import (
    PCOSDataset,
    HFVisionDataset,
    create_label_mapping,
    stratified_split_by_pid,
    stratified_pid_kfold,
)
from utils.transform import get_transform
from torchvision import transforms


class PCOSKFoldTrainer:
    def __init__(
        self,
        model_name: str,
        model_cache_dir: str,
        data_root_dir: str,
        label_path: str,
        result_root_dir: str,
        num_epochs: int = 5,
        learning_rate: float = 5e-5,
        optim_name: str = "adamw_torch",      # "grokadamw", "stable_adamw", "apollo_adamw", "adamw_torch"...
        weight_decay: float = 0.01,
        lr_scheduler_type: str = "cosine",    # "linear", "cosine", "constant_with_warmup" ...
        warmup_ratio: float = 0.1,
        fp16: bool = False,
        bf16: bool = False,
        grad_accum_steps: int = 1,
        optim_target_modules=None,            # APOLLO / GaLore용
        batch_size: int = 16,
        n_splits: int = 5,
        gpu_id: int = 0,
        distributed: bool = False,                # <- Multi-GPU
        logging_backend: str = "tensorboard",     # <- "wandb", "tensorboard", "none"
        wandb_project: str = "pcos-ultrasound",   # <- wandb 사용 시 project명
    ):
        """
        PCOS Ultrasound Classification K-Fold Trainer
        ==================================================
        Optional:
            distributed=True  → Multi-GPU DDP 활성화
            logging_backend="wandb" → wandb logging 사용
            logging_backend="tensorboard" → tensorboard 사용
        """

        # GPU 설정
        if not distributed:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 저장 인자들
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.data_root_dir = data_root_dir
        self.label_path = label_path
        self.result_root_dir = result_root_dir
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.optim_name = optim_name
        self.weight_decay = weight_decay
        self.lr_scheduler_type = lr_scheduler_type
        self.warmup_ratio = warmup_ratio
        
        self.fp16 = fp16
        self.bf16 = bf16
        if self.fp16 and self.bf16:
            raise ValueError("fp16 and bf16 cannot both be True. Choose only one.")

        self.grad_accum_steps = grad_accum_steps
        self.optim_target_modules = optim_target_modules
        self.batch_size = batch_size
        self.n_splits = n_splits

        self.distributed = distributed

        self.logging_backend = logging_backend
        self.wandb_project = wandb_project

        # Metrics
        self.accuracy = evaluate.load("accuracy")
        self.f1 = evaluate.load("f1")

        # Load CSV
        self.label_df = pd.read_csv(label_path)

        # Label mapping
        self.label_mapping = create_label_mapping(self.label_df, "label")

        # Transform
        self.train_tf, self.val_tf = get_transform(
            train_transform=[
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.RandomAffine(
                    degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)
                ),
                transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
            ]
        )

        # Train/Val/Test split
        self.train_df, self.val_df, self.test_df = stratified_split_by_pid(self.label_df)
        self.tune_df = pd.concat([self.train_df, self.val_df]).reset_index(drop=True)
        self.folds = stratified_pid_kfold(self.tune_df, n_splits=self.n_splits)

    # ------------------------
    # ----- Metrics ----------
    # ------------------------
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)

        return {
            "f1": self.f1.compute(references=labels, predictions=preds, average="macro")["f1"],
            "accuracy": self.accuracy.compute(references=labels, predictions=preds)["accuracy"],
        }

    # ------------------------
    # ----- 모델 초기화 -------
    # ------------------------
    def init_model(self):
        model = AutoModelForImageClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label_mapping),
            ignore_mismatched_sizes=True,
            cache_dir=self.model_cache_dir
        ).to(self.device)

        model.config.id2label = {int(v): str(k) for k, v in self.label_mapping.items()}
        model.config.label2id = {str(k): int(v) for k, v in self.label_mapping.items()}

        return model
    def hp_space(self, trial):
        # Optuna / WandB에서 조정할 파라미터 공간 정의
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64]),
            "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine"]),
        }

    def run_hpo_on_fold0(self, n_trials=20, backend="optuna"):
        # 0번 fold를 기준으로 HPO
        fold_train_df, fold_val_df = self.folds[0]

        train_base = PCOSDataset(
            fold_train_df,
            self.data_root_dir,
            filename_col="filename",
            label_col="label",
            label_mapping=self.label_mapping,
            transform=self.train_tf,
        )
        val_base = PCOSDataset(
            fold_val_df,
            self.data_root_dir,
            filename_col="filename",
            label_col="label",
            label_mapping=self.label_mapping,
            transform=self.val_tf,
        )

        train_dataset = HFVisionDataset(train_base)
        val_dataset   = HFVisionDataset(val_base)

        # 기본 arguments (여기 값들은 HPO에서 덮어씀)
        args = TrainingArguments(
            output_dir=f"{self.result_root_dir}/hpo_fold0",
            num_train_epochs=self.num_epochs,
            evaluation_strategy="epoch",
            save_strategy="no",       # HPO 단계에선 굳이 저장 X
            report_to="none",
        )

        model = self.init_model()

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )

        best_run = trainer.hyperparameter_search(
            direction="maximize",    # f1을 metric_for_best_model으로 쓴다고 가정
            hp_space=self.hp_space,
            n_trials=n_trials,
            backend=backend,         # "optuna", "wandb", "ray" 등
        )

        print("Best run:", best_run)
        print("Best hyperparameters:", best_run.hyperparameters)

        return best_run

    # ------------------------
    # ----- Fold Training -----
    # ------------------------
    def train_one_fold(self, fold_idx, fold_train_df, fold_val_df):
        print(f"\n========== Fold {fold_idx} Training ==========")

        # Dataset
        train_base = PCOSDataset(
            fold_train_df,
            self.data_root_dir,
            filename_col="filename",
            label_col="label",
            label_mapping=self.label_mapping,
            transform=self.train_tf,
        )
        val_base = PCOSDataset(
            fold_val_df,
            self.data_root_dir,
            filename_col="filename",
            label_col="label",
            label_mapping=self.label_mapping,
            transform=self.val_tf,
        )

        train_dataset = HFVisionDataset(train_base)
        val_dataset = HFVisionDataset(val_base)

        # 모델 초기화
        model = self.init_model()

        # Logging 선택
        report_to = []
        if self.logging_backend == "wandb":
            report_to = ["wandb"]
        elif self.logging_backend == "tensorboard":
            report_to = ["tensorboard"]
        else:
            report_to = ["none"]

        # TrainingArguments
        args = TrainingArguments(
            output_dir=f"{self.result_root_dir}/pcos_fold_{fold_idx}",
            learning_rate=self.learning_rate,
            optim=self.optim_name,                     # "grokadamw", "apollo_adamw", "stable_adamw"
            weight_decay=self.weight_decay,
            lr_scheduler_type=self.lr_scheduler_type, # "linear", "cosine", "constant_with_warmup"
            warmup_ratio=self.warmup_ratio,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=2,
            logging_dir=f"{self.result_root_dir}/logs_fold_{fold_idx}",
            report_to=report_to,
            run_name=f"pcos_fold_{fold_idx}" if self.logging_backend == "wandb" else None,
            ddp_find_unused_parameters=False,
            fp16=self.fp16,
            bf16=self.bf16,
            gradient_accumulation_steps=self.grad_accum_steps,
            optim_target_modules=self.optim_target_modules if self.optim_target_modules is not None else None,
            local_rank= int(os.getenv("LOCAL_RANK", -1)) if self.distributed else -1,
            ddp_backend="auto" if self.distributed else None
        )

        # Trainer 설정
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        metrics = trainer.evaluate()

        print(f"[Fold {fold_idx}] {metrics}")

        return metrics, model

    # ------------------------
    # ----- 전체 Fold 실행 ----
    # ------------------------
    def run_kfold(self):
        fold_results = []
        best_models = []

        for fold_idx, (fold_train_df, fold_val_df) in enumerate(self.folds):
            metrics, model = self.train_one_fold(fold_idx, fold_train_df, fold_val_df)
            fold_results.append(metrics)
            best_models.append(model)

        return fold_results, best_models

    # ------------------------
    # ----- Test Set 평가 -----
    # ------------------------
    def evaluate_test_for_fold(self, model, fold_idx):
        print(f"\n======= Evaluating Test Set (Fold {fold_idx}) =======")

        # Test dataset
        test_base = PCOSDataset(
            self.test_df,
            self.data_root_dir,
            filename_col="filename",
            label_col="label",
            label_mapping=self.label_mapping,
            transform=self.val_tf,
        )
        test_dataset = HFVisionDataset(test_base)

        trainer = Trainer(
            model=model,
            compute_metrics=self.compute_metrics
        )

        # Metrics
        metrics = trainer.evaluate(test_dataset)
        print(f"[Fold {fold_idx}] Test Metrics:", metrics)

        # Prediction
        raw_pred = trainer.predict(test_dataset)
        logits = raw_pred.predictions
        preds = np.argmax(logits, axis=1)

        df_result = pd.DataFrame({
            "filename": self.test_df["filename"].values,
            "label": self.test_df["label"].values,
            "pred": preds
        })

        # Save path (fold-aware)
        save_dir = os.path.join(self.result_root_dir, "test", f"fold_{fold_idx}")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, "test_results.csv")
        df_result.to_csv(save_path, index=False)

        print(f"[Saved] Fold {fold_idx} test results saved at: {save_path}")

        return metrics, df_result

    def evaluate_all_folds(self, models):
        print("\n======= Running Test Evaluation on ALL FOLDS =======")
        
        all_results = []
        
        for fold_idx, model in enumerate(models):
            metrics, df_result = self.evaluate_test_for_fold(model, fold_idx)
            all_results.append({
                "fold": fold_idx,
                "metrics": metrics
            })

        return all_results



# ----------------------------
# Example Usage
# ----------------------------
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PCOS KFold Training & Evaluation.")

    parser.add_argument('--model_name', type=str, default="google/vit-base-patch16-224")
    parser.add_argument('--model_cache_dir', type=str, default="/workspace/pcos_dataset/models")
    parser.add_argument('--data_root_dir', type=str, default="/workspace/pcos_dataset/Dataset")
    parser.add_argument('--label_path', type=str, default="/workspace/pcos_dataset/labels/기존_Dataset_info.csv")
    parser.add_argument('--result_root_dir', type=str, default="/workspace/pcos_dataset/results/baseline_기존라벨")
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--distributed', action='store_true', default=False, help='Enable multi-GPU distributed training')
    parser.add_argument('--logging_backend', type=str, default="tensorboard", choices=["wandb", "tensorboard", "none"])
    parser.add_argument('--wandb_project', type=str, default="pcos-ultrasound")

    args = parser.parse_args()

    trainer = PCOSKFoldTrainer(
        model_name=args.model_name,
        model_cache_dir=args.model_cache_dir,
        data_root_dir=args.data_root_dir,
        label_path=args.label_path,
        result_root_dir=args.result_root_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_splits=args.n_splits,
        gpu_id=args.gpu_id,
        distributed=args.distributed,
        logging_backend=args.logging_backend,
        wandb_project=args.wandb_project
    )

    fold_results, models = trainer.run_kfold()

    # 모든 fold 모델에 대해 test evaluation 실행
    test_results = trainer.evaluate_all_folds(models)
    print(test_results)
