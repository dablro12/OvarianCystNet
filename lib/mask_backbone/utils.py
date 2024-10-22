
import json
import csv
import pandas as pd 
import os 
import matplotlib.pyplot as plt
import numpy as np

def load_df(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    df = pd.DataFrame(data[1:], columns=data[0])
    return df

from PIL import Image
import cv2 
def load_img(img_path) -> Image:
    img = Image.open(img_path)
    return img


def load_json(json_path) -> dict:
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def _setup_coordinates(data_dict):
    coordinates = []
    
    # Check if 'shapes' exists and is not empty
    if 'shapes' not in data_dict or len(data_dict['shapes']) == 0:
        return []
    
    for shape in data_dict['shapes']:
        points = shape['points']  # This will be a list of points
        coordinates.extend(points)  # Add all points to the coordinates list
    
    return coordinates

def box_counter(data_dict, img_name):
    # Check if 'shapes' exists and is not empty
    if 'shapes' not in data_dict or len(data_dict['shapes']) == 0:
        print(f"No shapes found for {img_name}")
        return img_name
    
    # Ensure there are exactly 2 shapes to process
    if len(data_dict['shapes']) != 2:
        print(f"Incorrect number of shapes in {img_name}. Shapes found: {len(data_dict['shapes'])}")
        print(f"Shape data: {data_dict['shapes']}")
        print('-' * 100)
        return img_name

def comp_mask(img, bbox, mask, real_mask):
    # bbox 그리기
    x, y, w, h = bbox
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # SAM 마스크 합성
    colored_mask = np.zeros_like(img)
    colored_mask[mask > 0] = (0, 0, 255)  # 빨간색으로 SAM 마스크 칠하기
    img_with_mask = cv2.addWeighted(img, 0.7, colored_mask, 0.3, 0)
    
    # 실제 마스크 합성
    colored_real_mask = np.zeros_like(img)
    colored_real_mask[real_mask > 0] = (0, 255, 0)  # 초록색으로 실제 마스크 칠하기
    img_with_real_mask = cv2.addWeighted(img, 0.7, colored_real_mask, 0.3, 0)
    
    # 이미지 비교를 위한 플롯 생성
    plt.figure(figsize=(12, 6), dpi=256)

    # SAM 마스크 이미지
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_with_mask, cv2.COLOR_BGR2RGB))
    plt.title('SAM Mask + Image')
    plt.axis('off')

    # 실제 마스크 이미지
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_with_real_mask, cv2.COLOR_BGR2RGB))
    plt.title('Real Mask + Image')
    plt.axis('off')
    
    # 플롯 보여주기
    plt.show()

