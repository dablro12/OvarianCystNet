from torch.utils.data import Dataset, BatchSampler
import os 
from PIL import Image
import numpy as np
import torch 
import random
from torchvision import transforms
import albumentations as A
from torch.utils.data import DataLoader
class Custom_stratified_Dataset(Dataset):
    def __init__(self, df, root_dir, transform = None): #transform 가지고올거있으면 가지고 오기 
        self.df, self.root_dir = df, root_dir 
        self.transform = transform
        
    def __len__(self):
        return len(self.df) 
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.df['img_name'][idx] + '.png') 
        label = self.df['label'][idx]
        
        #이미지 open to PIL : pytorch는 PIL 선호
        image = Image.open(image_path).convert('RGB')
        
        # transform 
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_labels(self):
        return self.df['label']
    
class Custom_df_dataset(Dataset):
    def __init__(self, df, root_dir, transform = None):
        self.df = df
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for idx, row in self.df.iterrows():
            self.image_paths.append(os.path.join(root_dir, row['ID']+ '.png'))
            self.labels.append(row['label'])
    def get_labels(self):
        return self.labels
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        return image, label

class Custom_bus_dataset(Dataset):
    def __init__(self, df, root_dir, joint_transform=None):
        self.df = df
        self.joint_transform = joint_transform  # Add joint transform for image and mask
        self.image_paths = []
        self.mask_paths = []
        self.labels = []
        
        # Mapping for label encoding
        self.label_map = {'benign': 0, 'malignant': 1}
        
        for idx, row in self.df.iterrows():
            self.image_paths.append(os.path.join(root_dir, row['source'], 'dataset', f'{row["ID"]}.png'))
            ## >>>>> mask path settings
            # self.mask_paths.append(os.path.join(root_dir, row['source'], 'mask', f'{row["ID"]}.png')) # for oirignal mask
            self.mask_paths.append(os.path.join(root_dir, row['source'], 'mask-sam', f'{row["ID"]}.png')) #for med sam mask
            # self.mask_paths.append(os.path.join(root_dir, 'BK/mask-ft-medsam1', f'{row["ID"]}.png')) #for medsam1 finetuning mask
            # self.mask_paths.append(os.path.join(root_dir, 'BK/mask-ft-medsam2', f'{row["ID"]}.png')) #for medsam2 fintuning mask
            
            self.labels.append(self.label_map[row['label']])  # Encode 'benign' as 0, 'malignant' as 1
    
    def get_labels(self):
        return self.labels
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Keep mask as grayscale
        
        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)  # Apply joint transform to both

        return image, mask, torch.tensor(label)  # Convert label to tensor for model compatibility

class Custom_pcos_dataset(Dataset):
    def __init__(self, 
            df, 
            root_dir, 
            mask_use:bool, 
            class_num:int, 
            data_type:str,
            joint_transform=False, 
            torch_transform= False,
        ):
        self.df = df
        self.root_dir = root_dir
        self.mask_use = mask_use
        self.joint_transform = joint_transform
        self.torch_transform = torch_transform
        self.image_paths = []
        self.mask_paths = []
        # 모든 라벨을 담아둘 임시 리스트
        label_list = []
        # 파일 경로 및 라벨 구성
        for _, row in self.df.iterrows():
            # (1) 이미지 경로 세팅
            self.image_paths.append(
                os.path.join(root_dir, data_type, f'{row["ID"]}.png')
            )

            # (2) 마스크 경로 세팅 (mask_use=True일 때)
            if self.mask_use:
                self.mask_paths.append(
                    os.path.join(root_dir, 'mask', f'{row["ID"]}.png')
                )
            
            # (3) 라벨 전처리 및 label_list에 저장
            if class_num == 1:
                # 예시: 원본 라벨이 0이면 0, 1~2이면 1로 이진화
                if row['label'] == 0:
                    label_list.append(0)
                else:
                    label_list.append(1)
            else: # 3개 클래스 그대로 사용한다고 가정
                label_list.append(row['label'])# (4) label_list를 텐서로 변환해 보관
        
        # label_list 내부에 데이터가 숫자이면 tensor 변환
        if type(label_list[0]) == int:
            self.labels_tensor = torch.tensor(label_list, dtype=torch.long)

        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """ __getitem__에서는 실제 (image, mask, label)을 만듭니다. """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        # 마스크가 필요 없다면 빈 마스크를 리턴
        if self.mask_use:
            mask_path = self.mask_paths[idx]
            mask = Image.open(mask_path).convert('L')
        else:
            mask = Image.new('L', image.size)

        # Joint transform이 있다면 (image, mask) 같이 augment
        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)
        elif self.torch_transform:
            image = self.torch_transform(image)
            mask = self.torch_transform(mask)

        # 라벨 텐서에서 idx 위치의 값을 꺼냄
        label = self.labels_tensor[idx]
        return image, mask, label
class Custom_pcos_dataset_BERT(Dataset):
    def __init__(self, 
                 df, 
                 root_dir, 
                 mask_use: bool, 
                 class_num: int, 
                 data_type: str,
                 joint_transform=None, 
                 torch_transform=False):
        self._import_library()
        
        self.df = df
        self.root_dir = root_dir
        self.mask_use = mask_use
        self.joint_transform = joint_transform
        self.torch_transform = torch_transform
        self.image_paths = []
        self.mask_paths = []
        self.bert_label = bert_labeler(
            model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", 
            device="cuda:0"
        )
        label_list = []
        self.scalar_label_list = []  # 스칼라 라벨을 저장할 리스트

        # 파일 경로 및 라벨 구성
        for idx, row in self.df.iterrows():
            # 이미지 경로
            self.image_paths.append(os.path.join(root_dir, data_type, f'{row["ID"]}.png'))

            # 마스크 경로 (mask_use=True일 때)
            if self.mask_use:
                self.mask_paths.append(os.path.join(root_dir, 'mask', f'{row["ID"]}.png'))

            # 라벨 처리
            if class_num == 1:
                # 이진 분류 예시
                if row['label'] == 0:
                    label_list.append(0)
                else:
                    label_list.append(1)
                self.scalar_label_list.append(0 if row['label'] == 0 else 1)
            elif class_num == 768:  # BERT Embedding 사용 시
                if idx == 0:
                    print(f"Label Embedding Vector Preparation Start")
                    label_embeddings = self.prepare_embed_vector()
                
                if row['label'] == 0:
                    label_name = "a medical image indicating Benign condition"
                elif row['label'] == 1:
                    label_name = "a medical image indicating Borderline condition"
                elif row['label'] == 2:
                    label_name = "a medical image indicating Malignant condition"
                embedding_vector = label_embeddings[label_name]  # 사전 정의된 벡터 사용
                label_list.append(embedding_vector)
                self.scalar_label_list.append(row['label'])
            else:  # 3개 클래스 그대로 사용
                label_list.append(row['label'])
                self.scalar_label_list.append(row['label'])

        # 라벨 텐서 변환
        if isinstance(label_list[0], int):
            self.labels_tensor = torch.tensor(label_list, dtype=torch.long)
        else:  # BERT Embedding 사용 시
            self.labels_tensor = torch.tensor(np.array(label_list), dtype=torch.float32)
        
        # 스칼라 라벨 텐서 변환
        self.scalar_labels_tensor = torch.tensor(self.scalar_label_list, dtype=torch.long)
    def _import_library(self):
        from lib.datasets.bert import bert_labeler
        
    def prepare_embed_vector(self):
        """ Bert Embedding Model을 이용해 라벨 벡터를 미리 준비합니다. """
        label_names  = ["Benign", "Borderline", "Malignant"]
        label_names = ['a medical image indicating ' + label + ' condition' for label in label_names]
        label_embeddings = {}
        for label_name in label_names:
            embedding_vector = self.bert_label.encode(text=label_name)
            label_embeddings[label_name] = embedding_vector
        return label_embeddings

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """ __getitem__에서는 실제 (image, mask, label)을 만듭니다. """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        # 마스크가 필요 없다면 빈 마스크를 리턴
        if self.mask_use:
            mask_path = self.mask_paths[idx]
            mask = Image.open(mask_path).convert('L')
        else:
            mask = Image.new('L', image.size)

        # Joint transform이 있다면 (image, mask) 같이 augment
        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)
        elif self.torch_transform:
            image = self.torch_transform(image)
            mask = self.torch_transform(mask)

        # 라벨 텐서에서 idx 위치의 값을 꺼냄
        label = self.labels_tensor[idx]
        # 라벨 정규화
        label = label / label.norm()
        
        return image, mask, label, self.scalar_labels_tensor[idx]


class JointTransform:
    """
    의료영상에서도 사용할 수 있도록 개선된 JointTransform 예시입니다.
    - resize: (width, height)로 변경할 사이즈 (ex: (224, 224))
    - horizontal_flip, vertical_flip: 각각 True일 경우 50% 확률로 가로/세로 뒤집기
    - rotation: 정수 값. 예: 30 -> -30 ~ +30 범위 내 임의 회전
    - random_affine: True일 경우 RandomAffine 적용
        * random_affine_params로 degrees, shear, scale, translate 등 세부설정 가능
    - center_crop: (width, height). CenterCrop 적용
    - random_brightness: True일 경우 Albumentations RandomBrightnessContrast 적용
        * brightness_params로 p값 등 추가 설정 가능
    - normalize_mean, normalize_std: Normalize(mean, std)에 사용할 값
    (기본값은 ImageNet 통계, 의료영상이면 직접 구한 mean/std로 바꾸는 것을 권장)
    """
    def __init__(
        self,
        resize=None,
        center_crop = None,
        horizontal_flip=False,
        vertical_flip=False,
        rotation=False,
        random_brightness = False,
        random_affine = False,
        normalize_mean = [0.1663, 0.1663, 0.1663],
        normalize_std = [0.2037, 0.2037, 0.2037]
        ):
        self.resize = resize
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation = rotation
        self.center_crop = center_crop
        self.random_brightness = random_brightness
        self.random_affine = random_affine
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        
    def __call__(self, image, mask):
        # ---------- (1) Resize ----------
        if self.resize is not None:
            resize_transform = transforms.Resize(self.resize)
            image = resize_transform(image)
            mask = resize_transform(mask)
        
        # ---------- (6) Center Crop ----------
        if self.center_crop is not None:
            image = transforms.CenterCrop(self.center_crop)(image)
            mask = transforms.CenterCrop(self.center_crop)(mask)
        
        # ---------- (2) Horizontal flip ----------
        if self.horizontal_flip:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        # ---------- (3) Vertical flip ----------
        if self.vertical_flip:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)

        # ---------- (4) Random Rotation ----------
        if self.rotation:
            angle = random.randint(-self.rotation, self.rotation)
            image = transforms.functional.rotate(image, angle)
            mask = transforms.functional.rotate(mask, angle)

        # ---------- (5) Random Affine ----------
        if self.random_affine:
            image = transforms.RandomAffine(scale = (0.8, 1.2), degrees =0, shear = 0, translate= (0, 0.2))(image)
            mask = transforms.RandomAffine(scale = (0.8, 1.2), degrees =0, shear = 0, translate= (0, 0.2))(mask)
        
        # ---------- (7) Random Brightness (Albumentations) ----------
        if self.random_brightness:
            image = A.RandomBrightnessContrast(p=0.5)(
                image = np.array(image),
                brightness_limit = (-0.2, 0.2),
                contrast_limit = (-0.2, 0.2),
                p = 0.5,
            )['image'] # albumentation은 np.array로 받아야함
            image = Image.fromarray(image)

        # ---------- (8) ToTensor ----------
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        # ---------- (9) Normalize ----------
        if self.normalize_mean is not None and self.normalize_std is not None:
            image = transforms.Normalize(mean = self.normalize_mean, std = self.normalize_std)(image)
        return image, mask


##################################################################################################################################################

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform = None): #transform 가지고올거있으면 가지고 오기 
        self.root_dir = root_dir 
        self.transform = transform
        self.labels = []
        self.image_paths = []
        
        # data 읽어서 labels와 image_path에 저장
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for filename in os.listdir(label_dir):
                    file_path = os.path.join(label_dir, filename)
                    self.labels.append(float(label))
                    self.image_paths.append(file_path)
                
        
    def __len__(self):
        return len(self.labels) 
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        #이미지 open to PIL : pytorch는 PIL 선호
        # opencv bgr -> rgb 로 변환
        image = Image.open(image_path).convert('RGB')
        
        # 위아래 제외하고 crop 해놓기 -> 원본과 같은사이즈로
        # crop_img = image.crop((0, 120, image.width, image.height - 50))
        # image = Image.new("RGB", image.size, (0,0,0))
        # image.paste(crop_img, (0,150)) 
        
        # transform 
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_labels(self):
        return self.labels



    
class BalancedBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # class 에 따라 구분
        self.class0_indices = [i for i, (_, label) in enumerate(dataset) if label == 0]
        self.class1_indices = [i for i, (_, label) in enumerate(dataset) if label == 1]

    def __iter__(self):
        random.shuffle(self.class0_indices)
        random.shuffle(self.class1_indices)

        # 반반씩 뽑기 위한 배치 사이즈 조정
        half_batch = self.batch_size // 2

        for i in range(0, min(len(self.class0_indices), len(self.class1_indices)), half_batch):
            batch_indices = []
            batch_indices.extend(self.class0_indices[i:i + half_batch])
            batch_indices.extend(self.class1_indices[i:i + half_batch])
            random.shuffle(batch_indices)  # 배치 내부 셔플
            yield batch_indices

    def __len__(self):
        return min(len(self.class0_indices), len(self.class1_indices)) // (self.batch_size // 2)
    
#%% Imbalnaned Dataset
from sklearn.utils.class_weight import compute_class_weight

def get_class_weights(dataset):
    unique, cnts = np.unique(dataset.labels_tensor.numpy(), return_counts=True)
    weight = torch.tensor(cnts).float().sum() / torch.tensor(cnts).float()
    return weight / weight.sum()



def compute_mean_std(dataset, batch_size=32, num_workers=8):
    """
    dataset (torch.utils.data.Dataset): 이미 ToTensor()가 적용되는 Dataset
    """
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
    
    # 채널별 누적을 위한 변수 (float 로 해야 함)
    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)
    
    num_samples = 0
    
    for imgs, _, _ in loader:
        # imgs shape = (B, C, H, W)
        # B: batch_size
        # C: 채널 수 (3)
        
        batch_size_current = imgs.size(0)
        # (B, C, H, W) -> (C, B*H*W)
        imgs = imgs.view(batch_size_current, 3, -1)  # 예: (32, 3, 224*224)
        
        # channel-wise sum
        channel_sum += imgs.mean(dim=(0,2)) * batch_size_current  
        # channel-wise squared sum (분산 계산용)
        channel_squared_sum += (imgs ** 2).mean(dim=(0,2)) * batch_size_current
        
        num_samples += batch_size_current

    # 채널별 mean
    mean = channel_sum / num_samples
    # 채널별 표준편차 = sqrt(E[X^2] - (E[X])^2)
    # 여기서 E[X^2] = channel_squared_sum / num_samples
    mean_sq = channel_squared_sum / num_samples
    std = torch.sqrt(mean_sq - mean ** 2)
    
    return mean, std
        
        