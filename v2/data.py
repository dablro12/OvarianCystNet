from lib.dataset import data_split, k_fold_data_split
import pandas as pd 
from lib.sampler import BalancedBatchSampler
from lib.seed import seed_prefix
from lib.augmentation import SpeckleNoise
def workflow_data(datasheet_path, data_dir, sample_datasheet_path=None, sample_data_dir=None, binary_use = False, sampler = None, borderline_use = True):
    data_df = pd.read_csv(datasheet_path)
    train_df, _ = data_split(data_df, split_num = 5)
    train_df, valid_df = data_split(train_df, split_num = 5)
    
    if sample_datasheet_path is not None:
        sample_df = pd.read_csv(sample_datasheet_path)
        sample_df = sample_weight_distribution(sample_df, sample_weight=[0.1, 0.1, 0.1], seed=42)
    else:
        sample_df = None

    if borderline_use == False: # Borderline 제거하기
        df = df[df['label|0:양성, 1:중간형, 2:악성'] != 1]
        df = df.reset_index(drop=True)
        print(f"Borderline Use : {borderline_use}")
    
    train_loader = workflow_dataset(train_df, data_dir, sample_df= sample_df, sample_data_dir= sample_data_dir, type='train', sampler = sampler, binary_use= binary_use)
    valid_loader = workflow_dataset(valid_df, data_dir, type='valid', binary_use= binary_use)
    return train_loader, valid_loader

def workflow_k_fold_data(datasheet_path, data_dir, sample_datasheet_path=None, sample_data_dir=None, binary_use = False, sampler=None, n_splits=5, fdx:int = 0, borderline_use = True):
    """
        workflow_k_fold_data : K-Fold 데이터 분할 후 DataLoader 반환하는 함수
        (변경점) 기존 workflow_data 함수에서 K-Fold 분할을 추가한 함수
    """
    df = pd.read_csv(datasheet_path)
    if sample_datasheet_path is not None:
        sample_df = pd.read_csv(sample_datasheet_path)
        sample_df = sample_weight_distribution(sample_df, sample_weight=[0.1, 0.1, 0.1], seed=42)
    else:
        sample_df = None
    
    if borderline_use == False: # Borderline 제거하기
        df = df[df['label|0:양성, 1:중간형, 2:악성'] != 1]
        df = df.reset_index(drop=True)
        print(f"Borderline Use : {borderline_use}")
    
    # 미리 5개 폴드로 분할
    df, _ = data_split(df, split_num = 5)
    folds = k_fold_data_split(df, n_splits=n_splits)
    
    # 예: 폴드 0을 test, 폴드 1을 valid, 나머지를 train
    valid_idx = folds[fdx]
    train_idx = [idx for i, fold in enumerate(folds) if i not in [fdx] for idx in fold]
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    valid_df = df.iloc[valid_idx].reset_index(drop=True)
    train_loader = workflow_dataset(train_df, data_dir, sample_df = sample_df, sample_data_dir = sample_data_dir, type='train', bs = 24, sampler=sampler, binary_use = binary_use)
    valid_loader = workflow_dataset(valid_df, data_dir, type='valid', bs = 24, binary_use = binary_use)
    return train_loader, valid_loader


from lib.dataset import PCOS_Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.transforms import v2

def workflow_dataset(df:pd.DataFrame, data_dir:str, sample_df=None, sample_data_dir=None, type:str = 'train', bs:int = 24, sampler:str = None, binary_use:bool = False, ) -> DataLoader:
    if type == 'train':
        dataset = PCOS_Dataset(
            data_filenames = df['filename'],
            data_dir_path  = data_dir,
            labels         = df['label|0:양성, 1:중간형, 2:악성'],
            sample_data_filenames = sample_df['filename'] if sample_df is not None else None,
            sample_data_dir_path = sample_data_dir if sample_data_dir is not None else None,
            sample_labels  = sample_df['label|0:양성, 1:중간형, 2:악성'] if sample_df is not None else None,
            binary_use = binary_use,
            transform = v2.Compose([
                # v2.Resize((296, 296)), # 먼저 296x296으로 Resize
                # v2.CenterCrop(252),           # 224x224 중앙 자르기 -> 0.7977
                v2.RandomResizedCrop(224, scale = (0.8 ,1.2)),           # 224x224 렌담 중앙 줌인/줌아웃 자르기 -> 0.8089
                # Augmenttation 추가
                # RandomEqualize(p=0.5),    # Histogram Equalized
                # v2.RandomRotation(degrees = 15), # 랜덤 회전
                # v2.RandomHorizontalFlip(p = 0.5),    # 랜덤 수평 뒤집기
                # v2.RandomVerticalFlip(p = 0.5), # 랜덤 수직 뒤집기
                # Default Augmentation
                v2.Grayscale(num_output_channels=3),  # 3채널 회색변환 (RGB 형태 유지)
                v2.TrivialAugmentWide(), # [ADD] TrivialAugmentWide : A Simple Method for Improved Robustness and Calibration
                v2.ToTensor(),                # 텐서 변환 
                # v2.RandomApply([v2.GaussianNoise(mean = 0, sigma = 0.1, clip = True)], p=0.5), # 가우시안 노이즈
                # v2.RandomApply([SpeckleNoise(noise_level = 0.05)], p =0.5), # [Refer] SpeckleNoise : Automatic ovarian tumors recognition system based on ensemble convolutional neural network with ultrasound imaging
                # v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5.))], p=0.5), # 가우시안 Blur], p =0.5),
                # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 정규화 -> 0.7119
            ])
        )
        #%% Sampler 적용 여부에 따라 DataLoader 생성
        if sampler == 'balanced':
            loader = DataLoader(dataset, 
                    batch_size = bs, 
                    sampler = BalancedBatchSampler(
                        dataset = dataset,
                        labels = dataset.labels,
                        batch_size = bs,
                        ),
                    pin_memory=True,
                    pin_memory_device="cuda:0",
                    persistent_workers=True,
                    prefetch_factor= 4, 
                    num_workers= 8,
                    )
        else:
            loader = DataLoader(dataset, batch_size = bs, shuffle = True, pin_memory=True, pin_memory_device= "cuda:0", persistent_workers=True, prefetch_factor = 4, num_workers= 8, )
            
    else:
        dataset = PCOS_Dataset(
            data_filenames = df['filename'],
            data_dir_path  = data_dir,
            labels         = df['label|0:양성, 1:중간형, 2:악성'],
            sample_data_filenames = sample_df['filename'].reset_index(drop=True) if sample_df is not None else None,
            sample_data_dir_path = sample_data_dir if sample_data_dir is not None else None,
            sample_labels = sample_df['label|0:양성, 1:중간형, 2:악성'].reset_index(drop=True) if sample_df is not None else None,
            binary_use = binary_use,
            transform = v2.Compose([
                v2.Resize((296, 296)), # 먼저 296x296으로 Resize
                v2.CenterCrop(252),
                v2.Resize((224, 224)), # 먼저 296x296으로 Resize
                v2.Grayscale(num_output_channels=3),
                v2.ToTensor(),
            ])
        )
        loader = DataLoader(dataset, batch_size = bs, shuffle = False, pin_memory=True, pin_memory_device= "cuda:0", persistent_workers=True, prefetch_factor = 4, num_workers= 8, )
    return loader


def sample_weight_distribution(sample_df:pd.DataFrame, sample_weight:list = [1.0, 1.0, 1.0], seed:int = 42) -> pd.DataFrame:
    """
    3개의 클래스에 대해 얼마나 샘플링을 할지 결정하는 함수
    sample_weight index 순서대로 0, 1, 2 클래스에 대해 얼마나 가져올지에 대한 가중치를 의미
    """
    
    # sample_df에서 'label|0:양성, 1:중간형, 2:악성' 컬럼을 기준으로 각 클래스별로 샘플링할 개수를 결정
    class_0_df = sample_df[sample_df['label|0:양성, 1:중간형, 2:악성'] == 0]
    class_1_df = sample_df[sample_df['label|0:양성, 1:중간형, 2:악성'] == 1]
    class_2_df = sample_df[sample_df['label|0:양성, 1:중간형, 2:악성'] == 2]   
    
    
    # 각 클래스별로 샘플링할 개수를 가중치를 곱하여 결정
    # 가중치가 1이면 원래 개수만큼 샘플링
    class_0_sample_num = int(len(class_0_df) * sample_weight[0])
    class_1_sample_num = int(len(class_1_df) * sample_weight[1])
    class_2_sample_num = int(len(class_2_df) * sample_weight[2])
    
    # 랜덤으로 가져오기
    class_0_df = class_0_df.sample(n = class_0_sample_num, random_state = seed)
    class_1_df = class_1_df.sample(n = class_1_sample_num, random_state = seed)
    class_2_df = class_2_df.sample(n = class_2_sample_num, random_state = seed)
    
    # 샘플링한 데이터프레임을 합쳐서 반환
    sample_weight_df = pd.concat([class_0_df, class_1_df, class_2_df], axis = 0)
    sample_weight_df = sample_weight_df.reset_index(drop=True)
    print(f"Sampling Result : {len(sample_weight_df)}")
    return sample_weight_df
    