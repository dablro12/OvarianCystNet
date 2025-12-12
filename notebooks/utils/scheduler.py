import torch
class WarmupCosineAnnealingWarmRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Warmup + CosineAnnealingWarmRestarts 결합 스케줄러
    warmup_steps 동안 선형 증가 → 이후 CosineAnnealingWarmRestarts 적용
    """

    def __init__(
        self,
        optimizer,
        warmup_steps,
        T_0,
        T_mult=1,
        eta_min=0,
        last_epoch=-1,
    ):
        self.warmup_steps = warmup_steps
        self.wrapped = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
        )
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # warmup 비율
            warmup_ratio = float(self.last_epoch + 1) / float(self.warmup_steps)
            return [
                base_lr * warmup_ratio
                for base_lr in self.base_lrs
            ]
        else:
            # WarmRestarts 스케줄러에서 LR 계산
            return self.wrapped.get_lr()

    def step(self, epoch=None):
        if self.last_epoch < self.warmup_steps:
            super().step(epoch)
        else:
            # warmup 이후는 내부 Cosine scheduler 를 구동
            self.wrapped.step(epoch)
            # 부모 step도 호출하지만 내부 영향 없음
            super().step(epoch)
