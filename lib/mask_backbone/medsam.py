import os,sys
sys.path.append('../')
import cv2
from reference.MedSAM.segment_anything import sam_model_registry
from reference.MedSAM.utils.demo_tuning import BboxPromptDemo
import numpy as np
import pandas as pd  # 추가: 데이터프레임 로드를 위해 필요
from utils import *

def main():
    csv_path = "/mnt/hdd/octc/BACKUP/BreastUS/experiment/test_df_bbox.csv"
    root_dir = "/mnt/hdd/octc/BACKUP/BreastUS"
    MedSAM_CKPT_PATH = "/mnt/hdd/octc/experiment/medsam/weights/medsam_vit_b.pth"
    device = "cuda:0"
    ## Model Loader 
    medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()

    # CSV 파일에서 데이터프레임 로드
    data_df = pd.read_csv(csv_path)

    for idx, row in enumerate(data_df.iterrows()):
        file_name, label, bbox, data_type = row[1]['ID'] + '.png', row[1]['label'], row[1]['BBOX'], row[1]['source']
        img_path = os.path.join(root_dir, data_type, 'dataset', file_name)
        real_mask_path = os.path.join(root_dir, data_type, 'mask', file_name)
        mask_save_path = os.path.join(root_dir, data_type, 'mask-sam', file_name)
        
        img = cv2.imread(img_path)
        real_mask = cv2.imread(real_mask_path, cv2.IMREAD_GRAYSCALE)
        
        # BBOX 파싱 및 변환
        bbox = row[1]['BBOX'].replace('[', '').replace(']', '').split(',')
        bbox = [int(b) for b in bbox]
        
        # 모델을 사용한 세그멘테이션
        demo = BboxPromptDemo(medsam_model)
        demo.set_image_path(img_path)
        bboxes = [[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]]
        
        mask = demo.get_segmentations(bboxes, expand_ratio=0)[0]
        
        # mask를 이미지에 연하게 그리기 (mask와 img 합성)
        mask = mask.astype(np.uint8)
        
        # 마스크 저장
        mask = mask * 255
        cv2.imwrite(mask_save_path, mask)

        comp_mask(img, bbox, mask, real_mask)
        if idx == 5:
            break  # 5번째 이미지까지 실행 후 종료

if __name__ == "__main__":
    main()