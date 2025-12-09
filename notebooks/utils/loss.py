from torch.nn import CrossEntropyLoss
from transformers import Trainer, TrainerCallback
import torch
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


class WeightedLossTrainer(Trainer):

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

        if self.class_weights == "auto":
            cb = AutoWeightCallback()
            self.add_callback(cb)
            cb.trainer = self   # Trainer 객체 자동 주입


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")

        if self.class_weights is None or \
           (self.class_weights == "auto" and not isinstance(self.class_weights, torch.Tensor)):
            loss_fct = CrossEntropyLoss()
        else:
            loss_fct = CrossEntropyLoss(weight=self.class_weights.to(labels.device))

        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss
