# TSNE 
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

def scale_to_01_range(x):
    """데이터를 [0, 1] 범위로 스케일링합니다."""
    value_range = (np.max(x) - np.min(x))
    if value_range == 0:
        return x - np.min(x)
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range


def detect_outliers_isolation_forest(tsne_result, contamination=0.01):
    """Isolation Forest를 사용한 이상값 탐지."""
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(tsne_result)
    outliers = np.where(preds == -1)[0]
    return outliers


def tsne_dataloader(df, dataset_root_dir):
    """
    DataFrame에서 이미지 파일을 불러와 t-SNE에 사용할 데이터를 생성합니다.
    
    Parameters:
        df (pd.DataFrame): 'img_name' 열을 포함하는 DataFrame.
        
    Returns:
        data (np.ndarray): 평탄화된 이미지 배열.
        labels (np.ndarray): 각 이미지의 레이블 배열 (선택 사항).
    """
    data = []
    labels = []
    for _, row in df.iterrows():
        img_path = os.path.join(dataset_root_dir, str(row['ID']) + '.png')
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # 이미지 크기 조정
            img = cv2.resize(img, (256, 256))
            data.append(img.flatten())
            if 'label' in row:
                labels.append(row['label'])
        else:
            print(f"Warning: Image {img_path} could not be loaded.")
    data = np.array(data)
    labels = np.array(labels) if labels else None
    return data, labels

def tsne_plot(data, labels=None, n_components=2, perplexity=30, random_state=627):
    """
    t-SNE를 적용하고 결과를 시각화합니다.
    
    Parameters:
        data (np.ndarray): 고차원 데이터.
        labels (np.ndarray or list, optional): 시각화 시 색상을 구분할 레이블.
        n_components (int): t-SNE 임베딩 차원 (2 또는 3).
        perplexity (float): t-SNE의 perplexity 파라미터.
        random_state (int): 재현성을 위한 랜덤 시드.
    """
    # t-SNE 모델 생성
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    
    # t-SNE 수행
    tsne_result = tsne.fit_transform(data)
    
    # 시각화를 위한 데이터프레임 생성
    df_tsne = pd.DataFrame()
    df_tsne['TSNE1'] = tsne_result[:, 0]
    if n_components >= 2:
        df_tsne['TSNE2'] = tsne_result[:, 1]
    if n_components == 3:
        df_tsne['TSNE3'] = tsne_result[:, 2]
    
    if labels is not None:
        # 레이블을 숫자로 변환
        label_mapping = {'benign': 0, 'malignant': 1}
        numeric_labels = [label_mapping[label] for label in labels]
        df_tsne['Label'] = numeric_labels
    else:
        df_tsne['Label'] = 'All'
    
    # 시각화
    if n_components == 2:
        plt.figure(figsize=(10, 8))
        if labels is not None:
            sns.scatterplot(
                x='TSNE1', y='TSNE2',
                hue='Label',
                palette=sns.color_palette("hsv", len(np.unique(numeric_labels))),  # 자동으로 다양한 색상 할당
                data=df_tsne,
                alpha=0.7
            )
        else:
            sns.scatterplot(
                x='TSNE1', y='TSNE2',
                hue='Label',
                palette=['green'],  # 레이블이 없을 때 초록색
                data=df_tsne,
                legend=False,
                alpha=0.5
            )
        plt.title("t-SNE Visualization")
        plt.xlabel("TSNE1")
        plt.ylabel("TSNE2")
        if labels is not None:
            plt.legend(title='Class')
        plt.show()
        plt.close()
        
    elif n_components == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        if labels is not None:
            scatter = ax.scatter(
                df_tsne['TSNE1'], df_tsne['TSNE2'], df_tsne['TSNE3'],
                c=numeric_labels,
                cmap='viridis',  # 숫자 레이블을 색상으로 매핑
                alpha=0.7
            )
            # 범례 추가
            handles, _ = scatter.legend_elements(prop="colors")
            ax.legend(handles, ['benign', 'malignant'], title='Class')
        else:
            ax.scatter(
                df_tsne['TSNE1'], df_tsne['TSNE2'], df_tsne['TSNE3'],
                c='green',  # 레이블이 없을 때 초록색
                alpha=0.5
            )
        ax.set_title("t-SNE 3D Visualization")
        ax.set_xlabel("TSNE1")
        ax.set_ylabel("TSNE2")
        ax.set_zlabel("TSNE3")
        plt.show()
        plt.close()
    else:
        raise ValueError("n_components must be either 2 or 3.")
    
    # 이상값 탐지 (Isolation Forest 사용)
    outliers = detect_outliers_isolation_forest(tsne_result, contamination=0.01)
    
    # 이상값 결과 출력
    print(f"Number of outliers: {len(outliers)}")
    print(f"Outlier indices: {outliers}")
