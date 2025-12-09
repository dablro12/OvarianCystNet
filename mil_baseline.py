# mil_baseline.py
import sys 
sys.path.append("/workspace/notebooks")

import os
import math
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ---- custom imports ----
from utils.dataset import (
    PCOSMILDataset,
    create_label_mapping,
    stratified_split_by_pid,
    stratified_pid_kfold,
)
from utils.transform import get_transform
from utils.models import AttentionMIL, TransformerMIL

from torchvision import transforms


# ==========================================================
# Utility Metrics (HG Tuning Version)
# ==========================================================
def compute_metrics(y_true, y_pred, y_logit):
    """
    HG-style full metric set for MIL baseline:
    accuracy, macro/micro F1, precision, recall,
    Cohen's kappa, MCC, confusion matrix, ROC-AUC
    """

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

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_logit = np.array(y_logit)

    metrics = {}

    # 기본 분류 metric
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")
    metrics["f1_micro"] = f1_score(y_true, y_pred, average="micro")

    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["precision_micro"] = precision_score(y_true, y_pred, average="micro", zero_division=0)

    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_micro"] = recall_score(y_true, y_pred, average="micro", zero_division=0)

    # 추가 통계 metric
    metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
    metrics["matthews_corrcoef"] = matthews_corrcoef(y_true, y_pred)

    # confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    # ROC-AUC
    try:
        if len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = roc_auc_score(y_true, y_logit[:, 1])
        else:
            metrics["roc_auc_ovr"] = roc_auc_score(y_true, y_logit, multi_class="ovr", average="macro")
    except Exception:
        metrics["roc_auc"] = float("nan")

    return metrics

# ==========================================================
# PCOSKFoldMILTrainer CLASS
# ==========================================================
class PCOSKFoldMILTrainer:
    def __init__(
        self,
        data_root_dir,
        label_path,
        result_dir,
        model_type="transformer",
        embed_dim=256,
        transformer_depth=4,
        num_epochs=10,
        lr=1e-4,
        weight_decay=0.05,
        warmup_ratio=0.1,
        label_smoothing=0.1,
        n_splits=5,
        num_workers=16,
        gpu_id=0,
        logging_backend="tensorboard",
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data_root_dir = data_root_dir
        self.label_path = label_path
        self.result_dir = result_dir
        os.makedirs(result_dir, exist_ok=True)

        self.model_type = model_type
        self.embed_dim = embed_dim
        self.transformer_depth = transformer_depth
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.label_smoothing = label_smoothing
        self.n_splits = n_splits
        self.num_workers = num_workers
        self.logging_backend = logging_backend

        # Load labels
        self.df = pd.read_csv(label_path)
        self.label_mapping = create_label_mapping(self.df, "label")
        self.num_classes = len(self.label_mapping)

        # Augmentation
        self.train_tf, self.val_tf = get_transform(
            train_transform=[
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(10),
            ]
        )

    
        # Split PID-level
        train_df, val_df, test_df = stratified_split_by_pid(self.df)
        print("--------------------------------")
        print("Train labels:", train_df['label'].unique())
        print("Val labels:", val_df['label'].unique())
        print("Test labels:", test_df['label'].unique())
        print("--------------------------------")
        self.tune_df = pd.concat([train_df, val_df]).reset_index(drop=True)
        self.test_df = test_df

        self.folds = stratified_pid_kfold(self.tune_df, n_splits=n_splits)

    # --------------------------------------------------------
    # Model builder
    # --------------------------------------------------------
    def build_model(self):
        if self.model_type == "attention":
            return AttentionMIL(
                num_classes=self.num_classes,
                embed_dim=self.embed_dim,
            ).to(self.device)

        return TransformerMIL(
            num_classes=self.num_classes,
            embed_dim=self.embed_dim,
            depth=self.transformer_depth,
        ).to(self.device)
    # --------------------------------------------------------
    # Warmup + Cosine Scheduler
    # --------------------------------------------------------
    def build_scheduler(self, optimizer, total_steps):

        warmup_steps = int(total_steps * self.warmup_ratio)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # --------------------------------------------------------
    # Train One Epoch
    # --------------------------------------------------------
    def train_one_epoch(self, model, loader, optimizer, criterion, scheduler):
        model.train()
        running_loss = 0

        for batch in loader:
            imgs = batch["images"].to(self.device)
            label = batch["label"].long().to(self.device)

            optimizer.zero_grad()

            logits = model(imgs)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            loss = criterion(logits, label)
            loss.backward()

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        return running_loss / len(loader)
    # --------------------------------------------------------
    # Validation (for selecting best model — BUT NOT SAVED)
    # --------------------------------------------------------
    def validate(self, model, loader, criterion):
        model.eval()

        losses, preds, trues, logits_all = [], [], [], []

        with torch.no_grad():
            for batch in loader:
                imgs = batch["images"].to(self.device)
                label = batch["label"].long().to(self.device)

                logits = model(imgs)
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)

                loss = criterion(logits, label)

                losses.append(loss.item())
                preds.append(torch.argmax(logits, dim=1).cpu().item())
                trues.append(label.cpu().item())
                logits_all.append(logits.cpu().numpy())

        logits_all = np.vstack(logits_all)
        return np.mean(losses), compute_metrics(trues, preds, logits_all)



    # --------------------------------------------------------
    # Train Fold
    # --------------------------------------------------------
    def train_fold(self, fold_idx, train_df, val_df):
        print(f"\n========== Fold {fold_idx} ==========")

        # train/logs directory
        train_dir = f"{self.result_dir}/train_fold_{fold_idx}"
        logs_dir  = f"{self.result_dir}/logs_fold_{fold_idx}"
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        writer = SummaryWriter(log_dir=logs_dir)

        train_dataset = PCOSMILDataset(train_df, self.data_root_dir, transform=self.train_tf, label_mapping=self.label_mapping)
        val_dataset   = PCOSMILDataset(val_df, self.data_root_dir, transform=self.val_tf, label_mapping=self.label_mapping)

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=self.num_workers)
        val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)

        model = self.build_model()
        criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        total_steps = len(train_loader) * self.num_epochs
        scheduler = self.build_scheduler(optimizer, total_steps)

        best_f1 = -1
        best_state = None

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")

            train_loss = self.train_one_epoch(model, train_loader, optimizer, criterion, scheduler)
            val_loss, val_metrics = self.validate(model, val_loader, criterion)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Accuracy: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1_macro']:.4f}")

            writer.add_scalar("Train/Loss", train_loss, epoch)
            writer.add_scalar("Val/Loss", val_loss, epoch)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
            writer.add_scalar("Val/F1", val_metrics["f1_macro"], epoch)
            writer.add_scalar("Val/Accuracy", val_metrics["accuracy"], epoch)
            writer.add_scalar("Val/Precision", val_metrics["precision_macro"], epoch)
            writer.add_scalar("Val/Recall", val_metrics["recall_macro"], epoch)
            writer.add_scalar("Val/Cohen's Kappa", val_metrics["cohen_kappa"], epoch)
            writer.add_scalar("Val/Matthews Correlation Coefficient", val_metrics["matthews_corrcoef"], epoch)
            writer.add_scalar("Val/ROC-AUC", val_metrics["roc_auc"], epoch)

            if val_metrics["f1_macro"] > best_f1:
                best_f1 = val_metrics["f1_macro"]
                best_state = model.state_dict()
                torch.save(best_state, f"{train_dir}/best_model.pt")

        writer.close()
        return best_state

    # --------------------------------------------------------
    # Test Evaluation
    # --------------------------------------------------------
    def evaluate_test(self, state_dict, fold_idx):
        print(f"\n======= Test Evaluation Fold {fold_idx} =======")

        test_dir = f"{self.result_dir}/test_fold_{fold_idx}"
        os.makedirs(test_dir, exist_ok=True)

        model = self.build_model()
        model.load_state_dict(state_dict)
        model.eval()

        test_dataset = PCOSMILDataset(self.test_df, self.data_root_dir, transform=self.val_tf)
        test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

        preds, trues, logits_all, filenames = [], [], [], []

        with torch.no_grad():
            for idx, batch in enumerate(test_loader):

                fname = batch["filename"]
                filenames.append(f'{fname[0][0].split("_")[1]}')

                imgs  = batch["images"].to(self.device)
                label = batch["label"].long().to(self.device)

                logits = model(imgs)
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)

                preds.append(torch.argmax(logits, dim=1).cpu().item())
                trues.append(label.cpu().item())
                logits_all.append(logits.cpu().numpy())

        logits_all = np.vstack(logits_all)
        metrics = compute_metrics(trues, preds, logits_all)

        df = pd.DataFrame({
            "filename": filenames,
            "label": trues,
            "pred": preds
        })
        df.to_csv(f"{test_dir}/test_results.csv", index=False)
        pd.DataFrame([metrics]).to_csv(f"{test_dir}/test_metrics.csv", index=False)

        print(metrics)
        return metrics




    # --------------------------------------------------------
    # Run K-Fold
    # --------------------------------------------------------
    def run_kfold(self):
        all_results = []

        for fold_idx, (train_df, val_df) in enumerate(self.folds):
            best_state = self.train_fold(fold_idx, train_df, val_df)
            test_metrics = self.evaluate_test(best_state, fold_idx)
            all_results.append(test_metrics)

            torch.cuda.empty_cache()

        return all_results




# ==========================================================
# CLI
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root_dir", type=str, required=True)
    parser.add_argument("--label_path", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="transformer")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--gpu_id", type=int, default=0)

    args = parser.parse_args()

    trainer = PCOSKFoldMILTrainer(
        data_root_dir=args.data_root_dir,
        label_path=args.label_path,
        result_dir=args.result_dir,
        model_type=args.model_type,
        num_epochs=args.num_epochs,
        lr=args.lr,
        n_splits=args.n_splits,
        num_workers=args.num_workers,
        gpu_id=args.gpu_id,
    )

    results = trainer.run_kfold()
    print("Final Test Results:", results)
