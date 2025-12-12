import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers import Trainer, TrainerCallback
import numpy as np

class AutoWeightCallback(TrainerCallback):

    def on_init_end(self, args, state, control, **kwargs):
        """
        Trainer 초기화가 끝난 직후 호출됨.
        여기서 callback에 trainer 객체가 안전하게 주입됨.
        """
        self.trainer = kwargs["trainer"]   # ← 이제 trainer가 안전하게 들어옴


    def on_train_begin(self, args, state, control, **kwargs):
        trainer = self.trainer

        if trainer.class_weights != "auto":
            return

        labels = [int(item["labels"]) for item in trainer.train_dataset]

        labels = np.array(labels)
        _, counts = np.unique(labels, return_counts=True)

        weights = 1.0 / counts
        weights = weights / weights.sum() * len(counts)

        trainer.class_weights = torch.tensor(weights).float()

        print(f"[AutoWeightCallback] Computed weights: {trainer.class_weights.tolist()}")


class Poly1CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        epsilon: float = 1.0,
        reduction: str = "none",
        weight: Tensor = None,
    ):
        """
        Poly1 Cross-Entropy Loss (https://arxiv.org/abs/2104.10435)
        :param reduction: one of none|sum|mean
        """
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight

    def forward(self, logits, labels):
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).to(
            device=logits.device, dtype=logits.dtype
        )
        pt = torch.sum(labels_onehot * F.softmax(logits, dim=-1), dim=-1)
        ce = F.cross_entropy(
            input=logits, target=labels, reduction="none", weight=self.weight
        )
        loss = ce + self.epsilon * (1 - pt)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class Poly1FocalLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        epsilon: float = 1.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "none",
        weight: Tensor = None,
        pos_weight: Tensor = None,
        label_is_onehot: bool = False,
    ):
        """
        Poly1 Focal Loss (classification / segmentation)
        :param reduction: one of none|sum|mean
        """
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        self.label_is_onehot = label_is_onehot

    def forward(self, logits, labels):
        p = torch.sigmoid(logits)

        if not self.label_is_onehot:
            if labels.ndim == 1:
                labels = F.one_hot(labels, num_classes=self.num_classes)
            else:
                labels = (
                    F.one_hot(labels.unsqueeze(1), self.num_classes)
                    .transpose(1, -1)
                    .squeeze_(-1)
                )

        labels = labels.to(device=logits.device, dtype=logits.dtype)

        ce_loss = F.binary_cross_entropy_with_logits(
            input=logits,
            target=labels,
            reduction="none",
            weight=self.weight,
            pos_weight=self.pos_weight,
        )
        pt = labels * p + (1 - labels) * (1 - p)
        fl = ce_loss * ((1 - pt) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            fl = alpha_t * fl

        loss = fl + self.epsilon * torch.pow(1 - pt, self.gamma + 1)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class WeightedLossTrainer(Trainer):
    def __init__(
        self,
        *args,
        class_weights=None,
        model_init=None,
        loss_type: str = "poly1_ce",  # "ce" | "poly1_ce" | "poly1_focal"
        poly_epsilon: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        label_is_onehot: bool = False,
        **kwargs,
    ):
        # model_init 명시적으로 받기
        self.class_weights = class_weights
        self._model_init = model_init  # Trainer가 내부에서 사용하도록 저장
        self.loss_type = loss_type
        self.poly_epsilon = poly_epsilon
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_is_onehot = label_is_onehot

        # super 호출 시 model_init을 전달해야 함
        super().__init__(*args, model_init=model_init, **kwargs)

        # AUTO WEIGHT MODE
        if self.class_weights == "auto":
            cb = AutoWeightCallback()
            self.add_callback(cb)
            cb.trainer = self

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.get("logits")

        # prepare class weights on the correct device
        weight = (
            self.class_weights.to(labels.device)
            if isinstance(self.class_weights, torch.Tensor)
            else None
        )

        if self.loss_type == "poly1_focal":
            loss_fct = Poly1FocalLoss(
                num_classes=model.config.num_labels,
                epsilon=self.poly_epsilon,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                reduction="mean",
                weight=weight,
                label_is_onehot=self.label_is_onehot,
            )
        elif self.loss_type == "poly1_ce":
            loss_fct = Poly1CrossEntropyLoss(
                num_classes=model.config.num_labels,
                epsilon=self.poly_epsilon,
                reduction="mean",
                weight=weight,
            )
        else:  # plain CE
            loss_fct = CrossEntropyLoss(weight=weight)

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
