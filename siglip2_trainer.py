# siglip2_trainer.py
import os
import gc
import numpy as np
import pandas as pd
import torch

from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    AutoImageProcessor,
    SiglipForImageClassification,
)

from torchvision import transforms

from notebooks.utils.loss import WeightedLossTrainer
from notebooks.utils.dataset import (
    PCOSDataset,
    HFVisionDataset,
    create_label_mapping,
    stratified_split_by_pid,
    stratified_pid_kfold,
)
from notebooks.utils.transform import get_transform

def _safe_value(v):
    if isinstance(v, dict):
        v = list(v.values())[0]
    if isinstance(v, (np.float32, np.float64, np.int64, np.int32)):
        return float(v)
    if isinstance(v, list):
        return str(v)
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
        optim_name: str = "adamw_torch",
        weight_decay: float = 0.05,
        lr_scheduler_type: str = "cosine",
        warmup_ratio: float = 0.05,
        fp16: bool = False,
        bf16: bool = False,
        grad_accum_steps: int = 1,
        optim_target_modules=None,
        batch_size: int = 16,
        n_splits: int = 5,
        gpu_id: int = 0,
        distributed: bool = False,
        logging_backend: str = "tensorboard",
        wandb_project: str = "pcos-ultrasound",
        # SigLIP2 augment knobs (원하면 바꿔서 쓰세요)
        train_random_rotation_deg: int = 10,
        train_random_hflip_p: float = 0.5,
        train_random_sharpness: float = 2.0,
    ):
        if not distributed:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.data_root_dir = data_root_dir
        self.data_resize_size = data_resize_size
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

        # -------------------------
        # Load CSV / label mapping
        # -------------------------
        self.label_df = pd.read_csv(label_path)
        self.label_mapping = create_label_mapping(self.label_df, self.label_col_name)

        # -------------------------
        # SigLIP2 processor → mean/std/size
        # -------------------------
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_name,
            cache_dir=self.model_cache_dir,
        )


        # -------------------------
        # SigLIP2-style transforms
        # (MNIST 예제 구조 참고)
        # -------------------------
        self.train_tf, self.val_tf = get_transform(
            train_transform=[
                transforms.RandomHorizontalFlip(p=train_random_hflip_p),
                transforms.RandomRotation(train_random_rotation_deg),
            ],
            default_height_size=self.processor.size["height"],
            default_width_size=self.processor.size["width"],
            image_mean=self.processor.image_mean,
            image_std=self.processor.image_std,
        )
        # -------------------------
        # Train/Val/Test split + KFold
        # -------------------------
        self.train_df, self.val_df, self.test_df = stratified_split_by_pid(
            self.label_df, label_col=self.label_col_name
        )
        self.tune_df = pd.concat([self.train_df, self.val_df]).reset_index(drop=True)
        self.folds = stratified_pid_kfold(
            self.tune_df, n_splits=self.n_splits, label_col=self.label_col_name
        )

    # ------------------------
    # Data collator (안전)
    # ------------------------
    @staticmethod
    def collate_fn(examples):
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
        labels = torch.tensor([ex["labels"] for ex in examples], dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels}

    # ------------------------
    # Metrics
    # ------------------------
    def compute_metrics(self, eval_pred):
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
            cohen_kappa_score,
            matthews_corrcoef,
        )

        logits = eval_pred.predictions
        labels = eval_pred.label_ids
        preds = np.argmax(logits, axis=1)

        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()

        is_binary = len(np.unique(labels)) == 2
        average_type = "binary" if is_binary else "macro"

        results = {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average=average_type),
            "precision": precision_score(labels, preds, average=average_type, zero_division=0),
            "recall": recall_score(labels, preds, average=average_type),
            "cohen_kappa": cohen_kappa_score(labels, preds),
            "matthews_corrcoef": matthews_corrcoef(labels, preds),
        }

        try:
            if is_binary:
                results["roc_auc"] = roc_auc_score(labels, probs[:, 1])
            else:
                results["roc_auc_ovr"] = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        except:
            results["roc_auc"] = float("nan")

        if is_binary:
            pos_prob = probs[:, 1]
            best_thr = 0.5
            best_f1 = -1.0

            thresholds = np.linspace(0, 1, 200)
            for thr in thresholds:
                thr_pred = (pos_prob >= thr).astype(int)
                f1_val = f1_score(labels, thr_pred, average="binary")
                if f1_val > best_f1:
                    best_f1 = f1_val
                    best_thr = thr

            results["best_threshold"] = float(best_thr)
            results["best_f1_at_threshold"] = float(best_f1)

            thr_pred = (pos_prob >= best_thr).astype(int)
            try:
                results["best_threshold_roc_auc"] = float(roc_auc_score(labels, thr_pred))
            except:
                results["best_threshold_roc_auc"] = float("nan")

        return results

    # ------------------------
    # Model init (SigLIP2)
    # ------------------------
    def init_model(self):
        model = SiglipForImageClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label_mapping),
            ignore_mismatched_sizes=True,
            cache_dir=self.model_cache_dir,
            use_safetensors=True,
        ).to(self.device)

        model.config.id2label = {int(v): str(k) for k, v in self.label_mapping.items()}
        model.config.label2id = {str(k): int(v) for k, v in self.label_mapping.items()}
        return model

    # ------------------------
    # HPO space
    # ------------------------
    def hp_space(self, trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 6e-5, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64]),
            "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine"]),
        }

    def run_hpo_on_fold0(self, n_trials=20, backend="optuna"):
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
        val_dataset = HFVisionDataset(val_base)

        args = TrainingArguments(
            output_dir=f"{self.result_root_dir}/hpo_fold0",
            num_train_epochs=self.num_epochs,
            evaluation_strategy="epoch",
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
        )

        trainer = WeightedLossTrainer(
            class_weights="auto",
            model_init=self.init_model,          # ✅ callable로 수정 (중요)
            loss_type="poly1_focal",
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self.collate_fn,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor,
        )

        best_run = trainer.hyperparameter_search(
            direction="maximize",
            hp_space=self.hp_space,
            n_trials=n_trials,
            backend=backend,
        )

        print("Best run:", best_run)
        print("Best hyperparameters:", best_run.hyperparameters)
        return best_run

    # ------------------------
    # Train one fold
    # ------------------------
    def train_one_fold(self, fold_idx, fold_train_df, fold_val_df):
        print(f"\n========== Fold {fold_idx} Training ==========")

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
        val_dataset = HFVisionDataset(val_base)

        model = self.init_model()

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
            save_total_limit=5,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            load_best_model_at_end=True,
            logging_dir=f"{self.result_root_dir}/logs_fold_{fold_idx}",
            report_to=report_to,
            run_name=f"train_fold_{fold_idx}" if self.logging_backend == "wandb" else None,
            fp16=self.fp16,
            bf16=self.bf16,
            gradient_accumulation_steps=self.grad_accum_steps,
            remove_unused_columns=False,
        )

        trainer = WeightedLossTrainer(
            class_weights="auto",
            model=model,
            loss_type="poly1_focal",
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self.collate_fn,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )

        trainer.train()

        val_metrics = trainer.evaluate()
        print(f"[Fold {fold_idx}] Val Metrics:", val_metrics)

        best_val_threshold = val_metrics.get("eval_best_threshold", float("nan"))

        test_metrics, _ = self.evaluate_test_for_fold(model, fold_idx, best_val_threshold)

        del trainer
        del model
        torch.cuda.empty_cache()
        gc.collect()

        return val_metrics, test_metrics

    # ------------------------
    # Run all folds
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
    # Evaluate test per fold
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

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=f"{self.result_root_dir}/test_fold_{fold_idx}",
                per_device_eval_batch_size=self.batch_size,
                dataloader_num_workers=8,
                dataloader_pin_memory=True,
                fp16=self.fp16,
                bf16=self.bf16,
                report_to="none",
                remove_unused_columns=False,
            ),
            data_collator=self.collate_fn,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor,
        )

        metrics = trainer.evaluate(test_dataset)
        print(f"[Fold {fold_idx}] Test Metrics:", metrics)

        raw_pred = trainer.predict(test_dataset)
        logits = raw_pred.predictions
        preds = np.argmax(logits, axis=1)
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()

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

        test_metrics_path = os.path.join(save_dir, "test_metrics.csv")
        metrics_clean = {k: _safe_value(v) for k, v in metrics.items()}
        pd.DataFrame([metrics_clean]).round(3).to_csv(test_metrics_path, index=False)

        print(f"[Saved] Fold {fold_idx} test results saved at: {save_path}")

        return metrics, df_result


# ----------------------------
# Example Usage
# ----------------------------
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PCOS KFold Training & Evaluation (SigLIP2).")

    parser.add_argument('--model_name', type=str, default="google/siglip2-base-patch16-224")
    parser.add_argument('--model_cache_dir', type=str, default="/workspace/pcos_dataset/models")
    parser.add_argument('--data_root_dir', type=str, default="/workspace/pcos_dataset/Dataset")
    parser.add_argument('--label_path', type=str, default="/workspace/pcos_dataset/labels/기존_Dataset_info.csv")
    parser.add_argument('--label_col_name', type=str, default="label")
    parser.add_argument('--result_root_dir', type=str, default="/workspace/pcos_dataset/results/siglip2_baseline")
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=6e-6)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--distributed', action='store_true', default=False)
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
        wandb_project=args.wandb_project,
    )

    val_results, test_results = trainer.run_kfold()
    print("Val results per fold:", val_results)
    print("Test results per fold:", test_results)

