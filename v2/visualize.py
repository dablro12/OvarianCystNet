import math
import os
import argparse
import cv2
import numpy as np
import torch
import timm

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def plot_roc_curve(true_labels, pred_probs, binary_use, save_path = False, class_names=None):
    """
    주어진 데이터에 대해 ROC curve를 그리고,
    최적 임계값(Youden's J statistic 기준) 지점에 빨간 점을 찍습니다.
    
    Parameters:
      true_labels (array-like): 실제 label. 
           - binary인 경우: 0 또는 1의 값
           - multi-class인 경우: 각 샘플에 대해 정수형 클래스 인덱스 (예: 0,1,2,...)
      pred_probs (array-like): 예측 확률.
           - binary인 경우: 양성 클래스의 확률 (shape: [N])
           - multi-class인 경우: 각 클래스에 대한 확률 (shape: [N, n_classes])
      binary_use (bool): True이면 이진 분류, False이면 다중 분류.
      class_names (list of str): multi-class인 경우, 각 클래스의 이름 리스트.
      
    Returns:
      optimal_threshold:
         - binary인 경우: 최적 임계값 (float)
         - multi-class인 경우: 각 클래스의 최적 임계값을 담은 dict {class_index: threshold, ...}
    """
    if binary_use:
        # Binary case
        fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
        roc_auc = auc(fpr, tpr)
        # Youden's J statistic: tpr - fpr 최대값을 기준으로 optimal threshold 선택
        J = tpr - fpr
        idx_opt = np.argmax(J)
        optimal_threshold = thresholds[idx_opt]
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
        plt.scatter(fpr[idx_opt], tpr[idx_opt], color='red', 
                    label=f'Optimal Threshold = {optimal_threshold:.2f}', zorder=10)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        title = f'ROC Curve'
        if class_names is not None and len(class_names) > 0:
            title += f' for {class_names[0]}'
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
        return optimal_threshold
    else:
        # Multi-class case
        true_labels = np.array(true_labels)
        pred_probs = np.array(pred_probs)
        n_classes = pred_probs.shape[1]
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]
            
        optimal_thresholds = {}
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            # 이진화: i번 클래스를 positive, 나머지를 negative로
            binary_true = (true_labels == i).astype(int)
            fpr, tpr, thresholds = roc_curve(binary_true, pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            
            # 최적 threshold 계산 (Youden's J statistic)
            J = tpr - fpr
            idx_opt = np.argmax(J)
            opt_thr = thresholds[idx_opt]
            optimal_thresholds[i] = opt_thr
            
            plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f}, thr={opt_thr:.2f})')
            plt.scatter(fpr[idx_opt], tpr[idx_opt], color='red', zorder=10)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Curves')
        plt.legend(loc='lower right')
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
        
        return optimal_thresholds
    
def visualize_confidence_scores(confidence_scores, labels, class_names, save_path="confidence_score_plot.png"):
    """
    테스트 데이터셋의 Confidence Score를 저장하고 시각화하는 함수.
    각 클래스별 평균 및 95% 신뢰구간을 시각화하여 신뢰도를 분석.
    
    :param confidence_scores: PyTorch Tensor (샘플 x 클래스) 또는 (샘플, )
    :param labels: 실제 라벨 (샘플, )
    :param class_names: 클래스 이름 리스트
    :param save_path: 저장할 파일 경로
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch

    # PyTorch Tensor -> NumPy 변환
    if isinstance(confidence_scores, torch.Tensor):
        confidence_scores = confidence_scores.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # 차원 확인 및 변환
    if len(confidence_scores.shape) == 1:
        confidence_scores = confidence_scores.reshape(-1, 1)

    # 이진 분류 또는 다중 분류에 따른 클래스 처리
    num_classes = len(class_names)

    # 입력 데이터 크기 확인 및 예외 처리
    if confidence_scores.shape[1] < num_classes:
        # 부족한 클래스를 0으로 채워서 처리
        padding = num_classes - confidence_scores.shape[1]
        confidence_scores = np.pad(confidence_scores, ((0, 0), (0, padding)), mode='constant', constant_values=0)
        print(f"Warning: Confidence scores have fewer columns ({confidence_scores.shape[1]}) than expected classes ({num_classes}). Padding with zeros.")

    # 시각화 설정
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("husl", num_classes)

    # 각 클래스별 분포 박스 플롯
    boxplot_data = []
    for i in range(num_classes):
        class_scores = confidence_scores[:, i]
        boxplot_data.append(class_scores)

    # 박스 플롯 생성
    plt.boxplot(boxplot_data, labels=class_names, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='black'),
                medianprops=dict(color="red"),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'))

    # 각 포인트 오버레이
    for i, class_scores in enumerate(boxplot_data):
        # 랜덤 지터 추가하여 겹침 방지
        x = np.random.normal(i+1, 0.04, size=len(class_scores))
        plt.scatter(x, class_scores, alpha=0.5, color=palette[i], edgecolors='black', linewidth=0.5)

    # 레이블 및 제목 설정
    plt.title("Confidence Scores Distribution by Class", fontsize=14)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Confidence Score", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 저장 및 표시
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
