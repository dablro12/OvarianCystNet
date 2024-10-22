import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import LabelBinarizer
def multi_classify_metrics(y_true, y_pred, y_prob, average='weighted'):
    """
    다중 클래스 분류 평가 지표를 계산합니다.
    
    Parameters:
    - y_true: 실제 레이블 (1차원 배열)
    - y_pred: 예측 레이블 (1차원 배열)
    - y_prob: 각 클래스에 대한 예측 확률 (2차원 배열, shape=(n_samples, n_classes))
    - average: 다중 클래스 평균 방법 ('weighted', 'macro', 'micro')
    
    Returns:
    - metrics: 계산된 지표를 포함한 딕셔너리
    """
    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    
    # Precision
    pre = precision_score(y_true, y_pred, average=average, zero_division=0)
    
    # Sensitivity (Recall)
    sen = recall_score(y_true, y_pred, average=average, zero_division=0)
    
    # F1-score
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # AUC 계산 (One-vs-Rest 방식)
    lb = LabelBinarizer()
    lb.fit(y_true)
    y_true_binarized = lb.transform(y_true)
    if y_true_binarized.shape[1] == 1:
        # 이진 분류의 경우
        y_true_binarized = np.hstack((1 - y_true_binarized, y_true_binarized))
    auc = roc_auc_score(y_true_binarized, y_prob, average=average, multi_class='ovr')
    
    # Specificity 계산 (각 클래스별 평균)
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]
    specificity_per_class = []
    for i in range(num_classes):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_per_class.append(specificity)
    
    # 평균 Specificity 계산
    specificity = np.mean(specificity_per_class)
    
    metrics = {
        'Accuracy': acc * 100,
        'Precision': pre * 100,
        'Sensitivity': sen * 100,
        'Specificity': specificity * 100,
        'F1-score': f1 * 100,
        'AUC': auc * 100
    }
    return metrics

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

def binary_classify_metrics(y_true, y_pred, y_prob, test_on='False'):
    """
    바이너리 분류 평가 지표를 계산합니다.

    Parameters:
    - y_true: 실제 레이블 (1차원 배열, 0과 1로 구성)
    - y_pred: 예측 레이블 (1차원 배열, 0과 1로 구성)
    - y_prob: 양성 클래스에 대한 예측 확률 (1차원 배열)

    Returns:
    - metrics: 계산된 지표를 포함한 딕셔너리
    """
    # 배열 형태 변환
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    y_prob = np.array(y_prob).flatten()

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    
    # Precision
    pre = precision_score(y_true, y_pred, zero_division=0)
    
    # Sensitivity (Recall)
    sen = recall_score(y_true, y_pred, zero_division=0)
    
    # F1-score
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # AUC
    auc = roc_auc_score(y_true, y_prob)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # 지표를 곱하지 않고 그대로 저장
    metrics = {
        'Accuracy': acc,
        'Precision': pre,
        'Sensitivity': sen,
        'Specificity': specificity,
        'F1-score': f1,
        'AUC': auc
    }
    
    # 필요 시에만 ROC 곡선 관련 정보 추가
    if test_on != 'False':
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        metrics.update({
            'FPR': fpr.tolist(),
            'TPR': tpr.tolist(),
            'Thresholds': thresholds.tolist()
        })
    
    return metrics
