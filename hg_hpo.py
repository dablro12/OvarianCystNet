# hg_hpo.py
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
from notebooks.utils.loss import WeightedLossTrainer
from notebooks.utils.dataset import (
    PCOSDataset,
    HFVisionDataset,
    create_label_mapping,
    stratified_split_by_pid,
    stratified_pid_kfold,
)
from notebooks.utils.transform import get_transform
from torchvision import transforms
def _safe_value(v):
    # dict → 첫 번째 값 추출
    if isinstance(v, dict):
        v = list(v.values())[0]

    # numpy → python 기본 타입으로 변환
    if isinstance(v, (np.float32, np.float64, np.int64, np.int32)):
        return float(v)

    # 리스트 → confusion matrix 같은 경우: 문자열로 변환
    if isinstance(v, list):
        return str(v)  # 또는 json.dumps(v) 저장해도 됨

    # float 변환 가능한 경우
    try:
        return float(v)
    except:
        return str(v)


class PCOSKFoldTrainer:
    def __init__(
        self,
        model_name: str,
        model_cache_dir: str,
        data_root_dir: str,
        label_path: str,
        label_col_name: str,
        result_root_dir: str,
        num_epochs: int = 5,
        learning_rate: float = 5e-5,
        optim_name: str = "adamw_torch",      # "grokadamw", "stable_adamw", "apollo_adamw", "adamw_torch"...
        weight_decay: float = 0.05,
        lr_scheduler_type: str = "cosine",    # "linear", "cosine", "constant_with_warmup" ...
        warmup_ratio: float = 0.2, # 초반 학습을 더 부드럽게 만들어 Loss Blow-Up방지
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
        self.label_col_name = label_col_name
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

        # Load CSV
        self.label_df = pd.read_csv(label_path)

        # Label mapping
        self.label_mapping = create_label_mapping(self.label_df, self.label_col_name)

        # Transform
        self.train_tf, self.val_tf = get_transform(
            train_transform=[
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(10),
            ]
        )

        # Train/Val/Test split
        self.train_df, self.val_df, self.test_df = stratified_split_by_pid(self.label_df, label_col=self.label_col_name)
        self.tune_df = pd.concat([self.train_df, self.val_df]).reset_index(drop=True)
        self.folds = stratified_pid_kfold(self.tune_df, n_splits=self.n_splits, label_col=self.label_col_name)

    # ------------------------
    # ----- Metrics ----------
    # ------------------------
    def compute_metrics(self, eval_pred):
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
            confusion_matrix,
            cohen_kappa_score,
            matthews_corrcoef
        )
        import numpy as np
        import torch

        logits = eval_pred.predictions
        labels = eval_pred.label_ids
        preds = np.argmax(logits, axis=1)

        # probability
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()

        # -------------------------
        # 🔹 binary vs multiclass 체크
        # -------------------------
        is_binary = len(np.unique(labels)) == 2
        average_type = "binary" if is_binary else "macro"

        # -------------------------
        # 🔹 공통 metrics 계산
        # -------------------------
        results = {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average=average_type),
            "precision": precision_score(labels, preds, average=average_type, zero_division=0),
            "recall": recall_score(labels, preds, average=average_type),
            "cohen_kappa": cohen_kappa_score(labels, preds),
            "matthews_corrcoef": matthews_corrcoef(labels, preds),
        }

        # -------------------------
        # 🔹 ROC-AUC 계산
        # -------------------------
        try:
            if is_binary:
                results["roc_auc"] = roc_auc_score(labels, probs[:, 1])
            else:
                results["roc_auc_ovr"] = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        except:
            results["roc_auc"] = float("nan")

        # -------------------------
        # 🔥 Binary일 때만 Best Threshold 탐색
        # -------------------------
        if is_binary:
            pos_prob = probs[:, 1]
            best_thr = 0.5
            best_f1 = -1

            thresholds = np.linspace(0, 1, 200)
            for thr in thresholds:
                thr_pred = (pos_prob >= thr).astype(int)
                f1_val = f1_score(labels, thr_pred, average="binary")

                if f1_val > best_f1:
                    best_f1 = f1_val
                    best_thr = thr

            # 기록
            results["best_threshold"] = float(best_thr)
            results["best_f1_at_threshold"] = float(best_f1)

            # threshold 기반 ROC-AUC (binary output 기반)
            thr_pred = (pos_prob >= best_thr).astype(int)
            try:
                thr_auc = roc_auc_score(labels, thr_pred)
            except:
                thr_auc = float("nan")

            results["best_threshold_roc_auc"] = float(thr_auc)

        return results


    # ------------------------
    # ----- 모델 초기화 -------
    # ------------------------
    def init_model(self):
        model = AutoModelForImageClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label_mapping),
            ignore_mismatched_sizes=True,
            cache_dir=self.model_cache_dir,
            use_safetensors=True
        ).to(self.device)

        model.config.id2label = {int(v): str(k) for k, v in self.label_mapping.items()}
        model.config.label2id = {str(k): int(v) for k, v in self.label_mapping.items()}

        return model
    def hp_space(self, trial):
        """
        Define the hyperparameter search space.
        - Optuna/Ray: receives an actual trial object.
        - WandB: receives None, so we must return a sweep config dict.
        """
        # WandB backend passes trial=None and expects a sweep configuration
        if trial is None:
            return {
                "method": "random",
                "metric": {"name": "eval/f1", "goal": "maximize"},
                "parameters": {
                    "learning_rate": {
                        "min": 1e-6,
                        "max": 1e-4,
                        "distribution": "log_uniform",
                    },
                    # 추가 파라미터는 필요 시 여기에 정의
                    # "weight_decay": {"min": 0.0, "max": 0.1},
                    # "warmup_ratio": {"min": 0.0, "max": 0.2},
                },
            }

        # Optuna/Ray trial flow
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            # "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
            # "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
            # "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine"]),
        }

    def run_hpo_on_fold0(self, n_trials=20, backend="optuna"):
        # 0번 fold를 기준으로 HPO
        fold_train_df, fold_val_df = self.folds[0]

        train_base = PCOSDataset(
            fold_train_df,
            self.data_root_dir,
            filename_col="filename",
            label_col=self.label_col_name,
            label_mapping=self.label_mapping,
            transform=self.train_tf,
        )
        val_base = PCOSDataset(
            fold_val_df,
            self.data_root_dir,
            filename_col="filename",
            label_col=self.label_col_name,
            label_mapping=self.label_mapping,
            transform=self.val_tf,
        )

        train_dataset = HFVisionDataset(train_base)
        val_dataset   = HFVisionDataset(val_base)
        # Logging 선택
        if self.logging_backend == "wandb":
            report_to = ["wandb"]
        elif self.logging_backend == "tensorboard":
            report_to = ["tensorboard"]
        else:
            report_to = ["none"]
            
        # 기본 arguments (여기 값들은 HPO에서 덮어씀)
        args = TrainingArguments(
            num_train_epochs=self.num_epochs,
            save_strategy="no",
            report_to=report_to,
            run_name=f"hpo_fold0_{self.logging_backend}",
            per_device_train_batch_size=self.batch_size,  # 기본값
            per_device_eval_batch_size=self.batch_size,
            dataloader_num_workers=16,
            dataloader_pin_memory=True,
            learning_rate=self.learning_rate,
            warmup_ratio=self.warmup_ratio,
            lr_scheduler_type=self.lr_scheduler_type,
            weight_decay=self.weight_decay,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_dir=f"{self.result_root_dir}/logs_hpo_fold0",
            fp16=self.fp16,
            bf16=self.bf16
        )
        # trainer = Trainer(
        trainer = WeightedLossTrainer(
            class_weights='auto',
            model_init=self.init_model,   # ← 함수 자체를 전달해야 함 (중요)
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
            label_col=self.label_col_name,
            label_mapping=self.label_mapping,
            transform=self.train_tf,
        )
        val_base = PCOSDataset(
            fold_val_df,
            self.data_root_dir,
            filename_col="filename",
            label_col=self.label_col_name,
            label_mapping=self.label_mapping,
            transform=self.val_tf,
        )

        train_dataset = HFVisionDataset(train_base)
        val_dataset   = HFVisionDataset(val_base)

        # 모델 초기화
        model = self.init_model()

        # Logging 선택
        if self.logging_backend == "wandb":
            report_to = ["wandb"]
        elif self.logging_backend == "tensorboard":
            report_to = ["tensorboard"]
        else:
            report_to = ["none"]

        args = TrainingArguments(
            output_dir=f"{self.result_root_dir}/train_fold_{fold_idx}",
            learning_rate=self.learning_rate,
            optim=self.optim_name,
            weight_decay=self.weight_decay,
            lr_scheduler_type=self.lr_scheduler_type,
            warmup_ratio=self.warmup_ratio,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            dataloader_num_workers=16,
            dataloader_pin_memory=True,
            num_train_epochs=self.num_epochs,
            eval_strategy="epoch",
            save_strategy="epoch",              
            save_total_limit = 5,
            metric_for_best_model="f1",
            logging_dir=f"{self.result_root_dir}/logs_fold_{fold_idx}",
            report_to=report_to,
            run_name=f"train_fold_{fold_idx}" if self.logging_backend == "wandb" else None,
            fp16=self.fp16,
            bf16=self.bf16,
            gradient_accumulation_steps=self.grad_accum_steps,
        )

        # trainer = Trainer(
        trainer = WeightedLossTrainer(
            class_weights='auto',   # ← None이면 기존 CE 
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )

        # ---------- Train ----------
        trainer.train()

        # ---------- Val 평가 ----------
        val_metrics = trainer.evaluate()
        print(f"[Fold {fold_idx}] Val Metrics:", val_metrics)

        best_val_threshold = val_metrics.get("eval_best_threshold", float("nan"))

        # ---------- Test 평가 (바로 실행) ----------
        test_metrics, _ = self.evaluate_test_for_fold(model, fold_idx, best_val_threshold)

        # ---------- 메모리 정리 ----------
        import gc
        del trainer
        del model
        torch.cuda.empty_cache()
        gc.collect()

        return val_metrics, test_metrics

    # ------------------------
    # ----- 전체 Fold 실행 ----
    # ------------------------
    def run_kfold(self):
        all_val_results = []
        all_test_results = []

        for fold_idx, (fold_train_df, fold_val_df) in enumerate(self.folds):
            print(f"\n===== Running Fold {fold_idx} =====")
            val_metrics, test_metrics = self.train_one_fold(fold_idx, fold_train_df, fold_val_df)

            all_val_results.append({"fold": fold_idx, "metrics": val_metrics})
            all_test_results.append({"fold": fold_idx, "metrics": test_metrics})

        return all_val_results, all_test_results

    # ------------------------
    # ----- Test Set 평가 -----
    # ------------------------
    def evaluate_test_for_fold(self, model, fold_idx, best_val_threshold=None):
        print(f"\n======= Evaluating Test Set (Fold {fold_idx}) =======")

        test_base = PCOSDataset(
            self.test_df,
            self.data_root_dir,
            filename_col="filename",
            label_col=self.label_col_name,
            label_mapping=self.label_mapping,
            transform=self.val_tf,
        )
        test_dataset = HFVisionDataset(test_base)

        # trainer = Trainer(
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=f"{self.result_root_dir}/test_fold_{fold_idx}",
                per_device_eval_batch_size=self.batch_size,
                dataloader_num_workers=8,
                dataloader_pin_memory=True,
                fp16=self.fp16,
                bf16=self.bf16,
                report_to="none"
            ),
            compute_metrics=self.compute_metrics
        )


        metrics = trainer.evaluate(test_dataset)
        print(f"[Fold {fold_idx}] Test Metrics:", metrics)

        raw_pred = trainer.predict(test_dataset)
        logits = raw_pred.predictions
        preds = np.argmax(logits, axis=1)
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()

        # 예측 클래스 확률 및 (binary일 경우) 양성 확률
        prob_pred = probs[np.arange(len(preds)), preds]
        prob_pos = probs[:, 1] if probs.shape[1] > 1 else prob_pred 

        df_result = pd.DataFrame({
            "filename": self.test_df["filename"].values,
            "label": self.test_df[self.label_col_name].values,
            "pred": preds,
            "prob_pred": prob_pred,
            "prob_pos": prob_pos,
            "best_val_threshold": best_val_threshold,
        })

        save_dir = os.path.join(self.result_root_dir, f"test_fold_{fold_idx}")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, "test_results.csv")
        df_result.to_csv(save_path, index=False)
        
        # test_metrics.csv 추가
        test_metrics_path = os.path.join(save_dir, "test_metrics.csv")
        metrics_clean = {k: _safe_value(v) for k, v in metrics.items()}

        pd.DataFrame([metrics_clean]).round(3).to_csv(test_metrics_path, index=False)

        print(f"[Saved] Fold {fold_idx} test results saved at: {save_path}")

        return metrics, df_result

# ----------------------------
# Example Usage
# ----------------------------
import argparse
import json 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PCOS KFold Training & Evaluation.")

    parser.add_argument('--model_name', type=str, default="google/vit-base-patch16-224")
    parser.add_argument('--model_cache_dir', type=str, default="/workspace/pcos_dataset/models")
    parser.add_argument('--data_root_dir', type=str, default="/workspace/pcos_dataset/Dataset")
    parser.add_argument('--label_path', type=str, default="/workspace/pcos_dataset/labels/통합_Dataset_info_binary.csv")
    parser.add_argument('--label_col_name', type=str, default="USG_Ontology")
    parser.add_argument('--result_root_dir', type=str, default="/workspace/pcos_dataset/results/hpo_search/binary/vit-base-patch16-224")
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=160)
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
        label_col_name=args.label_col_name,
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
    # HPO 실행
    best_run = trainer.run_hpo_on_fold0(n_trials=10, backend='optuna')
    hp = best_run.hyperparameters
    save_path = os.path.join(args.result_root_dir, "hpo_results.json")
    with open(save_path, "w") as f:
        json.dump(hp, f, indent=4)