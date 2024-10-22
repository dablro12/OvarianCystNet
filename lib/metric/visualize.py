import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import pi

# Slope chart (연결된 점 그래프) 그리기
def plot_slope_chart(df, metric):
    plt.figure(figsize=(10, 6))

    for model in df['Model'].unique():
        mask_value = df[(df['Model'] == model) & (df['Version'] == 'mask')][metric].values[0]
        origin_value = df[(df['Model'] == model) & (df['Version'] == 'origin')][metric].values[0]
        sammask_value = df[(df['Model'] == model) & (df['Version'] == 'sammask')][metric].values[0]

        plt.plot([0, 1, 2], [mask_value, origin_value, sammask_value], marker='o', label=model)
        plt.text(0, mask_value, f'{mask_value:.3f}', va='center', ha='right', fontsize=10, color='blue')
        plt.text(1, origin_value, f'{origin_value:.3f}', va='center', ha='left', fontsize=10, color='red')
        plt.text(2, sammask_value, f'{sammask_value:.3f}', va='center', ha='left', fontsize=10, color='green')

    plt.xticks([0, 1, 2], ['Mask', 'Origin', 'SamMask'], fontsize=12)
    plt.ylabel(f'{metric} Value')
    plt.title(f'{metric} Comparison between Mask, Origin, and SamMask for each Model', fontsize=14)
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.show()

# Grouped Bar Plot 그리기
def plot_grouped_bar(df, metric):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y=metric, hue='Version', data=df, palette='Set2')
    plt.title(f'{metric} Comparison between Mask, Origin, and SamMask', fontsize=14)
    plt.ylabel(f'{metric} Value')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

# 레이다 차트를 그리기 위한 함수 (연구 보고서용)
def plot_radar_chart(df, model_name):
    metrics = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-score', 'AUC']
    
    mask_row = df[(df['Model'] == model_name) & (df['Version'] == 'mask')].iloc[0][metrics]
    origin_row = df[(df['Model'] == model_name) & (df['Version'] == 'origin')].iloc[0][metrics]
    sammask_row = df[(df['Model'] == model_name) & (df['Version'] == 'sammask')].iloc[0][metrics]

    # 레이다 차트를 위한 준비 작업
    angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
    angles += angles[:1]

    mask_values = mask_row.tolist()
    origin_values = origin_row.tolist()
    sammask_values = sammask_row.tolist()

    mask_values += mask_values[:1]
    origin_values += origin_values[:1]
    sammask_values += sammask_values[:1]

    # 차트 크기 및 레이아웃 설정
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    # 각 축에 성능 지표 추가
    plt.xticks(angles[:-1], metrics, color='black', size=12, weight='bold')

    # 성능 비교 (mask, origin, sammask) - 색상과 스타일 조정
    ax.plot(angles, mask_values, linewidth=2, linestyle='solid', label='Mask', color='#1f77b4')
    ax.fill(angles, mask_values, '#1f77b4', alpha=0.2)

    ax.plot(angles, origin_values, linewidth=2, linestyle='solid', label='Origin', color='#ff7f0e')
    ax.fill(angles, origin_values, '#ff7f0e', alpha=0.2)

    ax.plot(angles, sammask_values, linewidth=2, linestyle='solid', label='SamMask', color='#2ca02c')
    ax.fill(angles, sammask_values, '#2ca02c', alpha=0.2)

    # 각 축의 범위를 0에서 1로 설정하여 성능 비교 명확히
    ax.set_ylim(0, 1)

    # 그리드와 눈금선 설정
    ax.yaxis.set_tick_params(labelsize=10)
    ax.grid(color='gray', linestyle='--', linewidth=0.5)

    # 중앙에 제목 설정
    plt.title(f'{model_name} - Radar Chart (Mask vs Origin vs SamMask)', size=16, color='black', y=1.1, weight='bold')

    # 범례 위치 및 스타일 조정
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1), fontsize=12, frameon=True, fancybox=True, shadow=True)

    # 차트 보여주기
    plt.show()

# 바이올린 플롯 그리기 함수
def plot_violin_plot(df, metric):
    plt.figure(figsize=(10, 6))
    
    # 바이올린 플롯 생성
    sns.violinplot(x='Model', y=metric, hue='Version', data=df, split=True, inner="quart", palette="Set2")
    
    # 플롯 꾸미기
    plt.title(f'{metric} Comparison between Mask, Origin, and SamMask', fontsize=14)
    plt.ylabel(f'{metric} Value')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()
