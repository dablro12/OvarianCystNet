import torch
from torch.nn import functional as F
from torch import nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        Focal Loss for multi-class classification.

        Args:
            alpha (list or tensor, optional): Class-wise weights. Default is None.
            gamma (float, optional): Focusing parameter. Default is 2.
            reduction (str, optional): Reduction method ('mean', 'sum', 'none'). Default is 'mean'.
        """
        super(FocalLoss, self).__init__()
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, torch.Tensor):
                self.alpha = alpha
            else:
                raise TypeError("Alpha should be a list, tuple, or torch.Tensor")
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass for Focal Loss.

        Args:
            inputs (tensor): Logits from the model (before softmax), shape (batch_size, num_classes).
            targets (tensor): Ground truth class indices, shape (batch_size).

        Returns:
            tensor: Computed Focal Loss.
        """
        # Compute the cross-entropy loss for each sample
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')  # Shape: (batch_size)

        # Compute pt (probability of the true class)
        pt = torch.exp(-CE_loss)  # Shape: (batch_size)

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            # Gather the alpha value for each target class
            alpha = self.alpha[targets]  # Shape: (batch_size)
            F_loss = alpha * (1 - pt) ** self.gamma * CE_loss  # Shape: (batch_size)
        else:
            F_loss = (1 - pt) ** self.gamma * CE_loss  # Shape: (batch_size)

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# 라벨 스무딩 함수
def smooth_labels(labels, epsilon):
    """
    라벨 스무딩을 적용한 새로운 라벨 반환.
    labels: 원본 라벨 (0 또는 1)
    epsilon: 스무딩 값 (예: 0.1)
    """
    return labels * (1 - epsilon) + epsilon / 2