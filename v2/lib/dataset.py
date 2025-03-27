

#%% [Function] PCOS Dataset 
from torch.utils.data import Dataset
from lib.augmentation import PairedAffineTransform
from PIL import Image
import os
import torch 
import random 

class PCOS_Dataset(Dataset):
    def __init__(self, data_filenames, data_dir_path, labels = None, sample_data_filenames = None, sample_data_dir_path = None, sample_labels = None, transform=None, binary_use = False, need_paths = False):
        # [DEF] data
        self.data_dir_path = data_dir_path
        self.data_filenames = data_filenames
        self.data_filepaths = [os.path.join(data_dir_path, filename +'.png') for filename in self.data_filenames]
        self.labels = labels
        print(f"[Alert] Sample Dataset Use : {True if sample_data_filenames is not None else False}")
        if sample_data_filenames is not None: # sample_data_filenames이 있는 경우 data_filenames에 추가
            self.data_filenames = data_filenames + sample_data_filenames
            self.sample_data_filepaths=  [os.path.join(sample_data_dir_path, filename +'.png') for filename in sample_data_filenames]
            self.data_filepaths = self.data_filepaths + self.sample_data_filepaths
            self.labels = labels + sample_labels
        
        self.binary_use = binary_use
        self.need_paths = need_paths
        
        self.transform = transform
    def __len__(self):
        return len(self.data_filenames)
    
    def __getitem__(self, idx):
        data = Image.open(self.data_filepaths[idx])
        label = self.labels[idx]  # label이 int/float/list/np.array 등일 수 있음
        
        if self.binary_use:# 단일 분류 모델인경우
            # before_datasheet.csv (기존에 있던 데이터시트) : 0 - 양성, 1 - 중간형, 2 - 악성
            # label = 1 if label == 2 else 0 # 보더라인을 양성에 붙힌 경우 : 0.75161
            label = 0 if label == 0 else 1 # 보더라인을 악성에 붙힌 경우 0 : 양성 / 1 : 악성 및 보더라인 # Recall : 0.2
            # label = 1 if label == 0 else 0 # 반대로 지정해보기 1 : 양성 / 0 : 악성 및 보더라인 # Recall : 0.8

            # datasheet.csv (새로 업데이트된 데이터시트) : 0 - 양성, 1 - 악성, 2 - 중간형
            # 다중 뷴류 문제라면 float 타입을 쓰는 경우가 많습니다. (원-핫이 아닌 class index라고 가정)
            label = torch.tensor(label, dtype=torch.float32)
        else:
            # 다중 뷴류 문제라면 long 타입을 쓰는 경우가 많습니다. (원-핫이 아닌 class index라고 가정)
            label = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            data = self.transform(data)  # data는 보통 Tensor로 변환됨
        
        if self.need_paths: # Test의 경우 filepath가 필요할 수 있음.
            return data, label, self.data_filepaths[idx]
        
        return data, label
#%% Ultrasound Image Synthesis
class PCOS_Syntheis_Dataset(Dataset):
    def __init__(self, df, data_filenames, data_dir_path, mask_dir_path, labels, transform=None, mask_transform=None, affine_transform_use=False, binary_use = False, need_paths = False):

        # [DEF] Group Data by PID
        self.grouped_df = df.groupby('PID').filter(lambda x: len(x) >= 1)
        self.grouped_df['Group Num'] = self.grouped_df.groupby('PID').ngroup().reset_index(drop = True)

        # [DEF] data 
        self.data_dir_path = data_dir_path
        self.mask_dir_path = mask_dir_path
        self.data_filenames = data_filenames
        self.data_filepaths = [os.path.join(data_dir_path, filename +'.png') for filename in self.data_filenames]
        # [DEF] Label 
        self.labels = labels
        self.binary_use = binary_use
        self.need_paths = need_paths
        
        self.transform = transform
        self.affine_transform_use = affine_transform_use
        self.paired_affine = PairedAffineTransform(degrees= 30, translate = 0.3, scale = (0.8, 1.2), shear = 10)
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.data_filenames)
    
    def __getitem__(self, idx):
        filename = os.path.basename(self.data_filepaths[idx]).split('.png')[0]

        # [Group Data]
        select_filename = get_random_filename_from_same_group(self.grouped_df, filename)
        group_data = Image.open(os.path.join(self.data_dir_path, select_filename + '.png'))
        group_mask = Image.open(os.path.join(self.mask_dir_path, select_filename + '.png'))

        # [Original Code]
        data = Image.open(self.data_filepaths[idx])
        mask = Image.open(os.path.join(self.mask_dir_path, filename + '.png'))
        label = self.labels[idx]  # label이 int/float/list/np.array 등일 수 있음
        
        if self.binary_use:# 단일 분류 모델인경우
            # before_datasheet.csv (기존에 있던 데이터시트) : 0 - 양성, 1 - 중간형, 2 - 악성
            # label = 1 if label == 2 else 0 # 보더라인을 양성에 붙힌 경우 : 0.75161
            label = 0 if label == 0 else 1 # 보더라인을 악성에 붙힌 경우 0 : 양성 / 1 : 악성 및 보더라인 # Recall : 0.2
            # label = 1 if label == 0 else 0 # 반대로 지정해보기 1 : 양성 / 0 : 악성 및 보더라인 # Recall : 0.8

            # datasheet.csv (새로 업데이트된 데이터시트) : 0 - 양성, 1 - 악성, 2 - 중간형
            # 다중 뷴류 문제라면 float 타입을 쓰는 경우가 많습니다. (원-핫이 아닌 class index라고 가정)
            label = torch.tensor(label, dtype=torch.float32)
        else:
            # 다중 뷴류 문제라면 long 타입을 쓰는 경우가 많습니다. (원-핫이 아닌 class index라고 가정)
            label = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            data = self.transform(data)  # data는 보통 Tensor로 변환됨
            group_data = self.transform(group_data)
            
        if self.mask_transform:
            mask = self.mask_transform(mask)
            group_mask = self.mask_transform(group_mask)  # data는 보통 Tensor로 변환됨
        
        if self.affine_transform_use:
            # group_data와 group_mask 아핀변환을 동일하게 적용하기
            group_data, group_mask = self.paired_affine(group_data, group_mask)
            
        
        if self.need_paths: # Test의 경우 filepath가 필요할 수 있음.
            # return data, label, self.data_filepaths[idx]
            return data, mask, label, group_data, group_mask, self.data_filepaths[idx]
        
        # return data, label
        return data, mask, label, group_data, group_mask
#%% [Function] Get Random Filename
def get_random_filename_from_same_group(grouped_df, filename):
    """ 
    detail : grouped_df에서 특정 filename의 Group Num과 같은 Group Num을 가진 filename 중 하나를 출력하는 함수
    output : 만약 그룹 내에 같은 그룹의 다른 filename이 있으면 무작위 선택, 없으면 원래 filename 반환
    """
    # 주어진 filename의 Group Num 찾기
    group_num = grouped_df[grouped_df['filename'] == filename]['Group Num'].values[0]
    
    # 같은 Group Num을 가진 filename들 찾기 (원래 filename 제외)
    same_group_filenames = grouped_df[(grouped_df['Group Num'] == group_num) & 
                                      (grouped_df['filename'] != filename)]['filename'].tolist()
    
    # 같은 그룹의 다른 filename이 있으면 무작위 선택, 없으면 원래 filename 반환
    if same_group_filenames:
        return random.choice(same_group_filenames)
    else:
        return filename

#%% [Function] Data Split
"""
데이터셋을 분리할때 PID를 고려해서 처리하느 ㄴ함수 
"""
from sklearn.model_selection import StratifiedGroupKFold
def get_pid(item):
    return item.split('_')[1]

def data_split(df, split_num:int = 5):
    # Load Data
    # PID별로 데이터 그룹핑
    df['PID'] = df['filename'].apply(lambda x: x.split('_')[1])
    # StratifiedGroupKFold 객체 생성
    sgkf = StratifiedGroupKFold(n_splits= split_num, shuffle=True, random_state=int(os.getenv('SEED')))

    X = df['filename']  # 실제 모델 입력 feature는 다양할 수 있으나 여기서는 예시로 filename
    y = df['label|0:양성, 1:중간형, 2:악성']     # 분할 시 stratify 기준
    groups = df['PID']  # 그룹(동일 PID는 같은 세트로)

    # 한 번의 분리를 위해 첫 번째 split만 사용
    for train_idx, test_idx in sgkf.split(X, y, groups):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        break


    return train_df, test_df


def k_fold_data_split(df, n_splits=5):
    """
    전체 데이터를 n_splits개의 폴드로 분리한 후,
    각 폴드 인덱스를 딕셔너리 형태로 반환합니다.
    """
    df['PID'] = df['filename'].apply(lambda x: x.split('_')[1])
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state= int(os.getenv('SEED')))
    folds = []
    X = df['filename']
    y = df['label|0:양성, 1:중간형, 2:악성']
    groups = df['PID']
    
    for train_idx, test_idx in sgkf.split(X, y, groups):
        folds.append(test_idx)
    
    return folds



# #%% [Function] PCOS Dataset 
# from torch.utils.data import Dataset
# from PIL import Image
# import os
# import torch 
# class PCOS_Dataset(Dataset):
#     def __init__(self, data_filenames, data_dir_path, labels, transform=None, binary_use = False, need_paths = False):
#         # [DEF] data 
#         self.data_dir_path = data_dir_path
#         self.data_filenames = data_filenames
#         self.data_filepaths = [os.path.join(data_dir_path, filename +'.png') for filename in self.data_filenames]
#         # [DEF] Label 
#         self.binary_use = binary_use
#         self.need_paths = need_paths

#         self.labels = self.binary_use_option(labels, borderline_to_class = "malignant") # borderline 클래스를 악성으로 붙힌 경우
#         # self.labels = self.binary_use_option(labels, borderline_to_class = "benign") # borderline 클래스를 양성으로 붙힌 경우
#         # self.labels = self.binary_use_option(labels, borderline_to_class = "borderline to 1") # borderline 클래스만 1로 붙힌 경우
#         self.transform = transform
#     def __len__(self):
#         return len(self.data_filenames)
    
#     def __getitem__(self, idx):
#         data = Image.open(self.data_filepaths[idx])
#         label = self.labels[idx]  # label이 int/float/list/np.array 등일 수 있음
        
#         if self.binary_use:# 단일 분류 모델인경우
#             # 다중 뷴류 문제라면 float 타입을 쓰는 경우가 많습니다. (원-핫이 아닌 class index라고 가정)
#             label = torch.tensor(label, dtype=torch.float32)
#         else:
#             # 다중 뷴류 문제라면 long 타입을 쓰는 경우가 많습니다. (원-핫이 아닌 class index라고 가정)
#             label = torch.tensor(label, dtype=torch.long)
        
#         if self.transform:
#             data = self.transform(data)  # data는 보통 Tensor로 변환됨
        
#         if self.need_paths: # Test의 경우 filepath가 필요할 수 있음.
#             return data, label, self.data_filepaths[idx]
        
#         return data, label
    
#     def binary_use_option(self, labels, borderline_to_class: str):
#         """
#         before_datasheet.csv (0 - 양성, 1 - 중간형, 2 - 악성)
#         datasheet.csv         (0 - 양성, 1 - 악성, 2 - 중간형)

#         borderline_to_class 에 따라:
#         - "benign"          => label=2 를 1로, 나머지 0
#         - "malignant"       => label=0 을 0으로, 나머지 1
#         - "borderline to 1" => label=1 을 1로, 나머지 0
#         """
#         # 먼저 복사해서 경고 방지
#         labels = labels.copy()

#         if self.binary_use:
#             if borderline_to_class == "benign":
#                 # label=2 => 1, 그 외 => 0
#                 labels = (labels == 2).astype(int)
#             elif borderline_to_class == "malignant":
#                 # label=0 => 0, 그 외 => 1
#                 labels = (labels != 0).astype(int)
#             elif borderline_to_class == "borderline to 1":
#                 # label=1 => 1, 그 외 => 0
#                 labels = (labels == 1).astype(int)

#         return labels

            

# #%% [Function] Data Split
# """
# 데이터셋을 분리할때 PID를 고려해서 처리하느 ㄴ함수 
# """
# from sklearn.model_selection import StratifiedGroupKFold
# def get_pid(item):
#     return item.split('_')[1]

# def data_split(df, split_num:int = 5):
#     # Load Data
#     # PID별로 데이터 그룹핑
#     df['PID'] = df['filename'].apply(lambda x: x.split('_')[1])
#     # StratifiedGroupKFold 객체 생성
#     sgkf = StratifiedGroupKFold(n_splits= split_num, shuffle=True, random_state=int(os.getenv('SEED')))

#     X = df['filename']  # 실제 모델 입력 feature는 다양할 수 있으나 여기서는 예시로 filename
#     y = df['label|0:양성, 1:중간형, 2:악성']     # 분할 시 stratify 기준
#     groups = df['PID']  # 그룹(동일 PID는 같은 세트로)

#     # 한 번의 분리를 위해 첫 번째 split만 사용
#     for train_idx, test_idx in sgkf.split(X, y, groups):
#         train_df = df.iloc[train_idx].reset_index(drop=True)
#         test_df = df.iloc[test_idx].reset_index(drop=True)
#         break


#     return train_df, test_df


# def k_fold_data_split(df, n_splits=5):
#     """
#     전체 데이터를 n_splits개의 폴드로 분리한 후,
#     각 폴드 인덱스를 딕셔너리 형태로 반환합니다.
#     """
#     df['PID'] = df['filename'].apply(lambda x: x.split('_')[1])
#     sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state= int(os.getenv('SEED')))
#     folds = []
#     X = df['filename']
#     y = df['label|0:양성, 1:중간형, 2:악성']
#     groups = df['PID']
    
#     for train_idx, test_idx in sgkf.split(X, y, groups):
#         folds.append(test_idx)
    
#     return folds