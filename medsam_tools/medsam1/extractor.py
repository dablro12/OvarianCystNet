import os, sys 
sys.path.append('../')

import json 
# %% X,Y Coord Extract
def x_y_data_extrack(data_label_li):
    use_data_li =[]

    for file_path in data_label_li:
        with open(file_path, 'r') as f:
            json_data = json.load(f)

        xy_coord = {}
        for x_y_point in json_data['shapes']:
            # X,Y 두개 리스트로 정리된 좌표를 딕셔너리로 정리
            x_y_label = x_y_point['label']
            coords_li = []
            for point_li in x_y_point['points']:
                for point in point_li:
                    coords_li.append(point)
            # 좌표 정렬
            coords_li = sorted(coords_li)
            xy_coord[x_y_label] = coords_li
        # 'x', 'y' 두개가 있는 경우만 사용
        if 'x' in xy_coord.keys() and 'y' in xy_coord.keys():
            use_data_li.append(file_path.replace('json','png').replace('Labels', 'Dataset'))
    return use_data_li

def json_png_matching(data_li, json_li):
    """ data_label_li와 data_li에 매칭되는 파일에 대해서만 사용하기 위한 함수 """ 
    use_data_li = []
    for json_file in json_li:
        json_file = json_file.replace('Labels', 'Dataset').replace('json', 'png')
        if json_file in data_li:
            use_data_li.append(json_file)
    return use_data_li

def extract_xy_coords_from_json(file_path):
    """
    주어진 JSON 파일에서 X, Y 좌표를 추출하여 딕셔너리로 반환하는 함수.

    :param file_path: JSON 파일 경로
    :return: X, Y 좌표를 담은 딕셔너리
    """
    with open(file_path, 'r') as f:
        json_data = json.load(f)

    xy_coord = {}
    for x_y_point in json_data['shapes']:
        # X,Y 두개 리스트로 정리된 좌표를 딕셔너리로 정리
        x_y_label = x_y_point['label']
        coords_li = []
        for point_li in x_y_point['points']:
            for point in point_li:
                coords_li.append(point)
        # 좌표 정렬
        coords_li = sorted(coords_li)
        xy_coord[x_y_label] = coords_li
    
    return xy_coord

def extract_bbox_from_xy_coords(xy_coords):
    """
    주어진 X, Y 좌표를 이용하여 Bounding Box를 추출하는 함수.

    :param xy_coords: X, Y 좌표를 담은 딕셔너리
    :return: Bounding Box
    """
    x_coords = xy_coords['x']
    y_coords = xy_coords['y']
    # x, y 좌표를 이용하여 Bounding Box 계산
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    # like cv2
    bbox = [x_min, y_min, x_max, y_max]
    return bbox

def extract_bbox_from_json(file_path):
    """
    주어진 JSON 파일에서 Bounding Box를 추출하는 함수.

    :param file_path: JSON 파일 경로
    :return: Bounding Box
    """
    xy_coords = extract_xy_coords_from_json(file_path)
    bbox = extract_bbox_from_xy_coords(xy_coords)

    return bbox


import numpy as np
import os

join = os.path.join
import torch
from skimage import io, transform
from reference.MedSAM1.MedSAM_Inference import medsam_inference, show_box, show_mask

def init_medsam1(ckpt_path:str):
    from segment_anything import sam_model_registry
    
    MedSAM1_CKPT_PATH = ckpt_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    medsam1_model = sam_model_registry['vit_b'](checkpoint = MedSAM1_CKPT_PATH)
    medsam1_model = medsam1_model.to(device)
    medsam1_model.eval()
    return medsam1_model, device

def medsam1_inference_tool(model, img_path, label_path, device):
    # Image preprocessing - 이미 처리된 이미지와 바운딩 박스를 사용
    img_np = io.imread(img_path)
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = img_np
    H, W, _ = img_3c.shape
    # 이미지 리사이즈 및 정규화
    img_1024 = transform.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )  # normalize to [0, 1], (H, W, 3)

    # 이미지를 텐서로 변환
    img_1024_tensor = (
        torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )
    # bbox 처리 
    bbox = extract_bbox_from_json(label_path)

    # 바운딩 박스를 1024x1024 크기에 맞게 스케일 변환
    box_np = np.array([[int(x) for x in bbox]])  # bbox는 JSON으로부터 추출된 바운딩 박스
    box_1024 = box_np / np.array([W, H, W, H]) * 1024
    
    with torch.no_grad():
        image_embedding = model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)
        medsam_seg = medsam_inference(model, image_embedding, box_1024, H, W)

    return medsam_seg


import cv2
import pandas as pd

# %% Load Data 
pcos_data_dir = '/mnt/hdd/octc/BACKUP/Dataset'
pcos_label_dir = '/mnt/hdd/octc/BACKUP/Labels'

data_li = os.listdir(pcos_data_dir)
data_li = [os.path.join(pcos_data_dir, x) for x in data_li]
data_label_li = [os.path.join(pcos_label_dir, x) for x in os.listdir(pcos_label_dir)]

use_json_data_label_li = x_y_data_extrack(data_label_li) # x,y 좌표가 있는 데이터만 추출
use_data_li = json_png_matching(data_li, use_json_data_label_li) # x,y 좌표가 있는 데이터
print(f"처리 전 : {len(data_li)}개")
print(f"처리 후 : {len(use_data_li)}개")

save_dir = '/mnt/hdd/octc/BACKUP/MEDSAM1_Mask' # save dir settings !!

# %% Model Load
medsam1_model, device  = init_medsam1(ckpt_path = "../checkpoints/MedSAM1/medsam_vit_b.pth")

# %% Inference & Save
for img_path in use_data_li:
    label_path = img_path.replace('Dataset', 'Labels').replace('.png', '.json')
    img = cv2.imread(img_path)
    bbox = extract_bbox_from_json(label_path)
    
    # inference 
    medsam_seg = medsam1_inference_tool(medsam1_model, img_path, label_path, device)
    
    # save for binary mask (0,1) -> (0, 255)
    save_path = img_path.replace('Dataset', 'MEDSAM1_Mask').replace('.png', '.png')
    cv2.imwrite(save_path, medsam_seg*255)