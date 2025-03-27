import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, recall_score, precision_score, confusion_matrix
)
# 시각화시 한글 지원
plt.rc('font', family='NanumGothic')
plt.rc('axes', unicode_minus=False)

def calculate_metrics(labels, preds, binary_use, threshold = 0.5):
    """
    labels: numpy array or torch tensor (정답 라벨)
    preds: numpy array (예측 확률, shape: [N, num_classes])
    """
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy().flatten()
    if torch.is_tensor(preds):
        if binary_use:
            preds = preds.cpu().numpy().flatten()
        else:
            preds = preds.cpu().numpy()
    
    metrics = {}
    if binary_use:
        try:
            metrics['auc'] = float(roc_auc_score(labels, preds))
        except Exception as e:
            metrics['auc'] = 0.0
            
        pred_class = (preds >= threshold).astype(int)
        metrics['f1'] = f1_score(labels, pred_class)
        metrics['accuracy'] = accuracy_score(labels, pred_class)
        metrics['recall'] = recall_score(labels, pred_class)
    else:
        try:
            metrics['auc'] = float(roc_auc_score(labels, preds, multi_class='ovr', average='macro'))
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
                auc_i = float(roc_auc_score(y_true_i, y_score_i))
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



def plot_confusion_matrix_from_preds(labels, preds, binary_use, save_path, class_names=None, normalize=False, title='Confusion Matrix', threshold = 0.5):
    """
    labels: numpy array or torch tensor (정답 라벨)
    preds: numpy array or torch tensor (예측 확률), shape: [N, num_classes] (multi-class) 또는 [N] (binary)
    binary_use: 이진 분류 여부
    class_names: 클래스 이름 리스트 (예: ['class0', 'class1', ...])
    normalize: True이면 정규화된 값을 보여줌
    title: 플롯 제목
    """
    # 텐서이면 numpy로 변환
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy().flatten()
    if torch.is_tensor(preds):
        if binary_use:
            preds = preds.cpu().numpy().flatten()
        else:
            preds = preds.cpu().numpy()
    
    # 예측 클래스를 결정 (이진: 임계값 0.5, 다중: argmax)
    if binary_use:
        pred_class = (preds >= threshold).astype(int)
    else:
        pred_class = preds.argmax(axis=1)
    
    cm = confusion_matrix(labels, pred_class)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    # 클래스 이름이 주어졌다면 사용, 아니면 숫자 인덱스 사용
    if class_names is None:
        classes = np.arange(cm.shape[0])
    else:
        classes = class_names
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # 혼동 행렬 각 셀에 텍스트 추가
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    if save_path: # save_path가 주어진 경우 저장
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def calculate_metrics_v1(labels, preds, binary_use):
    """
    labels: numpy array or torch tensor (정답 라벨)
    preds: numpy array (예측 확률, shape: [N, num_classes])
    """
    # 텐서이면 numpy 배열로 변환 후 1차원으로 평탄화
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy().flatten()
    # preds의 경우, binary_use인 경우에만 flatten하고, 그렇지 않으면 그대로 유지합니다.
    if torch.is_tensor(preds):
        if binary_use:
            preds = preds.cpu().numpy().flatten()
        else:
            preds = preds.cpu().numpy()
    
    metrics = {}
    if binary_use:
        try:
            metrics['auc'] = float(roc_auc_score(labels, preds))
        except Exception as e:
            metrics['auc'] = 0.0
        pred_class = (preds >= 0.5).astype(int)
        metrics['f1'] = f1_score(labels, pred_class)
        metrics['accuracy'] = accuracy_score(labels, pred_class)
        metrics['recall'] = recall_score(labels, pred_class)
    else:
        try:
            metrics['auc'] = float(roc_auc_score(labels, preds, multi_class='ovr', average='macro'))
        except Exception as e:
            metrics['auc'] = 0.0
        preds_class = preds.argmax(axis=1)
        metrics['f1'] = f1_score(labels, preds_class, average='macro')
        metrics['accuracy'] = accuracy_score(labels, preds_class)
        metrics['recall'] = recall_score(labels, preds_class, average='macro')
    return metrics