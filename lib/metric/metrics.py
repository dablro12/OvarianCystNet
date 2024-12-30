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
    
    # AUC 계산
    auc = roc_auc_score(LabelBinarizer().fit_transform(y_true), y_prob, average=average)
    
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
    print("Computing Metrics Complete!!")
    return metrics

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

def multi_classify_metrics_v2(y_true, y_prob, optimize_thresholds=False):
    """
    다중 클래스 분류 모델의 예측 결과에 대한 평가 지표를 계산합니다.
    
    Parameters:
    - y_true: 실제 레이블 (각 샘플의 정답 클래스를 나타내는 리스트)
    - y_prob: 각 샘플마다 [클래스0 확률, 클래스1 확률, 클래스2 확률] 형태의 리스트
    - optimize_thresholds: 클래스별 임계값 최적화를 수행할지 여부 (기본값: False)
    
    Returns:
    - eval_dict (dict): 다음 내용을 포함하는 딕셔너리
        * acc : Accuracy
        * auc : AUC (One-vs-Rest, macro)
        * f1  : F1-score (macro)
        * conf_mat : 혼동행렬 (리스트 형태)
        * sensitivity : 각 클래스별 Sensitivity
        * specificity : 각 클래스별 Specificity
        * tp, tn, fp, fn : 각 클래스별 TP, TN, FP, FN
        * thresholds : 각 클래스별 최적 임계값 (임계값 최적화 시 포함)
    """
    
    from sklearn.preprocessing import label_binarize
    
    num_classes = y_prob.shape[1]
    
    # (2) 예측 라벨
    pred_label = np.argmax(y_prob, axis=1)

    # (3) Accuracy
    acc = accuracy_score(y_true, pred_label)

    # (4) AUC (One-vs-Rest, macro)
    auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')

    # (5) F1 (macro)
    f1 = f1_score(y_true, pred_label, average='macro')

    # (6) Confusion Matrix
    conf_mat = confusion_matrix(y_true, pred_label)
    total_samples = conf_mat.sum()

    # (7) Sensitivity & Specificity (per-class)
    sens_list, spec_list = [], []
    tp_list, tn_list, fp_list, fn_list = [], [], [], []

    for i in range(num_classes):
        TP = conf_mat[i, i]
        FN = conf_mat[i, :].sum() - TP
        FP = conf_mat[:, i].sum() - TP
        TN = total_samples - (TP + FN + FP)

        sensitivity = TP / (TP + FN) if (TP + FN) else 0
        specificity = TN / (TN + FP) if (TN + FP) else 0

        sens_list.append(sensitivity)
        spec_list.append(specificity)

        tp_list.append(TP)
        fn_list.append(FN)
        fp_list.append(FP)
        tn_list.append(TN)

    eval_dict = {
        "acc" : round(acc, 3),
        "auc" : round(auc, 3),
        "f1" : round(f1, 3),
        "conf_mat": conf_mat.tolist(),
        "sensitivity" : [round(i, 3) for i in sens_list],
        "specificity" : [round(i, 3) for i in spec_list],
        "tp" : tp_list,
        "tn" : tn_list,
        "fp" : fp_list,
        "fn" : fn_list
    }
    
    if optimize_thresholds:
        # 클래스별 임계값 최적화
        thresholds = []
        y_true_binarized = label_binarize(y_true, classes=list(range(num_classes)))
        
        for i in range(num_classes):
            fpr, tpr, thresh = roc_curve(y_true_binarized[:, i], y_prob[:, i])
            # 민감도(tpr)를 최대화하는 임계값 선택
            # 여기서는 TPR - FPR가 최대가 되는 지점을 선택 (Youden's J statistic)
            youdens_j = tpr - fpr
            optimal_idx = np.argmax(youdens_j)
            optimal_threshold = thresh[optimal_idx]
            thresholds.append(optimal_threshold)
        
        # 최적 임계값을 사용하여 예측 라벨 재생성
        pred_label_opt = []
        for prob in y_prob:
            class_preds = prob >= thresholds
            if np.any(class_preds):
                # 임계값을 만족하는 클래스 중 확률이 가장 높은 클래스를 선택
                chosen_class = np.argmax(prob * class_preds)
                pred_label_opt.append(chosen_class)
            else:
                # 모든 클래스가 임계값 미만인 경우 가장 높은 확률의 클래스 선택
                pred_label_opt.append(np.argmax(prob))
        
        # 새로운 Confusion Matrix 및 메트릭 계산
        conf_mat_opt = confusion_matrix(y_true, pred_label_opt)
        acc_opt = accuracy_score(y_true, pred_label_opt)
        auc_opt = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        f1_opt = f1_score(y_true, pred_label_opt, average='macro')

        sens_list_opt, spec_list_opt = [], []
        tp_list_opt, tn_list_opt, fp_list_opt, fn_list_opt = [], [], [], []

        for i in range(num_classes):
            TP = conf_mat_opt[i, i]
            FN = conf_mat_opt[i, :].sum() - TP
            FP = conf_mat_opt[:, i].sum() - TP
            TN = total_samples - (TP + FN + FP)

            sensitivity = TP / (TP + FN) if (TP + FN) else 0
            specificity = TN / (TN + FP) if (TN + FP) else 0

            sens_list_opt.append(sensitivity)
            spec_list_opt.append(specificity)

            tp_list_opt.append(TP)
            fn_list_opt.append(FN)
            fp_list_opt.append(FP)
            tn_list_opt.append(TN)

        eval_dict.update({
            "acc_opt" : round(acc_opt, 3),
            "auc_opt" : round(auc_opt, 3),
            "f1_opt" : round(f1_opt, 3),
            "conf_mat_opt": conf_mat_opt.tolist(),
            "sensitivity_opt" : [round(i, 3) for i in sens_list_opt],
            "specificity_opt" : [round(i, 3) for i in spec_list_opt],
            "tp_opt" : tp_list_opt,
            "tn_opt" : tn_list_opt,
            "fp_opt" : fp_list_opt,
            "fn_opt" : fn_list_opt,
            "thresholds" : [round(t, 3) for t in thresholds]
        })

    return eval_dict
#%% Binary 
def binary_classify_metrics(y_true, y_pred, y_prob, test_on=False):
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
    if test_on:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        metrics.update({
            'FPR': fpr.tolist(),
            'TPR': tpr.tolist(),
            'Thresholds': thresholds.tolist()
        })
    
    return metrics

def binary_classify_metrics_v2(y_true, y_prob, optimize_thresholds=False):
    """
    Computes evaluation metrics for binary classification models.

    Parameters:
    - y_true (array-like): Actual binary labels (0 or 1) for each sample.
    - y_prob (array-like): Predicted probabilities for the positive class (1) for each sample.
    - optimize_thresholds (bool): Whether to optimize the classification threshold based on Youden's J statistic.

    Returns:
    - eval_dict (dict): Dictionary containing the following keys:
        * acc: Accuracy
        * auc: AUC-ROC
        * f1: F1-score
        * conf_mat: Confusion matrix [TN, FP, FN, TP]
        * sensitivity: Sensitivity (Recall or True Positive Rate)
        * specificity: Specificity (True Negative Rate)
        * tp, tn, fp, fn: Counts of True Positives, True Negatives, False Positives, and False Negatives
        * thresholds (if optimize_thresholds=True): Optimal threshold for classification
        * acc_opt, auc_opt, f1_opt, conf_mat_opt, sensitivity_opt, specificity_opt (if optimize_thresholds=True)
    """
    
    # Ensure y_true and y_prob are numpy arrays
    y_true = np.array(y_true)
    y_prob = np.array(y_prob).flatten()
    
    # Check if y_prob is in correct range
    if not ((y_prob >= 0).all() and (y_prob <= 1).all()):
        raise ValueError("y_prob should contain probabilities between 0 and 1.")
    
    # (1) Predicted labels using default threshold of 0.5
    pred_label = (y_prob >= 0.5).astype(int)
    
    # (2) Accuracy
    acc = accuracy_score(y_true, pred_label)
    
    # (3) AUC-ROC
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        # Handle cases where only one class is present in y_true
        auc = float('nan')
    
    # (4) F1 Score
    f1 = f1_score(y_true, pred_label)
    
    # (5) Confusion Matrix
    # Confusion matrix returns in the order: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, pred_label)
    if cm.shape == (1, 1):
        # Handle cases with only one class present in y_true
        if y_true[0] == 1:
            tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
        else:
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
    else:
        tn, fp, fn, tp = cm.ravel()
    
    # (6) Sensitivity and Specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    eval_dict = {
        "acc": round(acc, 3),
        "auc": round(auc, 3) if not np.isnan(auc) else "nan",
        "f1": round(f1, 3),
        "conf_mat": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        },
        "sensitivity": round(sensitivity, 3),
        "specificity": round(specificity, 3),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }
    
    if optimize_thresholds:
        # Optimize threshold using Youden's J statistic
        fpr, tpr, thresh = roc_curve(y_true, y_prob)
        youdens_j = tpr - fpr
        optimal_idx = np.argmax(youdens_j)
        optimal_threshold = thresh[optimal_idx]
        
        # Apply optimal threshold
        pred_label_opt = (y_prob >= optimal_threshold).astype(int)
        
        # Recompute metrics with optimized threshold
        acc_opt = accuracy_score(y_true, pred_label_opt)
        
        try:
            auc_opt = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc_opt = float('nan')
        
        f1_opt = f1_score(y_true, pred_label_opt)
        
        cm_opt = confusion_matrix(y_true, pred_label_opt)
        if cm_opt.shape == (1, 1):
            # Handle cases with only one class present in y_true
            if y_true[0] == 1:
                tn_opt, fp_opt, fn_opt, tp_opt = 0, 0, 0, cm_opt[0, 0]
            else:
                tn_opt, fp_opt, fn_opt, tp_opt = cm_opt[0, 0], 0, 0, 0
        else:
            tn_opt, fp_opt, fn_opt, tp_opt = cm_opt.ravel()
        
        sensitivity_opt = tp_opt / (tp_opt + fn_opt) if (tp_opt + fn_opt) > 0 else 0
        specificity_opt = tn_opt / (tn_opt + fp_opt) if (tn_opt + fp_opt) > 0 else 0
        
        eval_dict.update({
            "threshold_opt": round(optimal_threshold, 3),
            "acc_opt": round(acc_opt, 3),
            "auc_opt": round(auc_opt, 3) if not np.isnan(auc_opt) else "nan",
            "f1_opt": round(f1_opt, 3),
            "conf_mat_opt": {
                "TN": int(tn_opt),
                "FP": int(fp_opt),
                "FN": int(fn_opt),
                "TP": int(tp_opt)
            },
            "sensitivity_opt": round(sensitivity_opt, 3),
            "specificity_opt": round(specificity_opt, 3)
        })
    
    return eval_dict