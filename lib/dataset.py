from torch.utils.data import Dataset, BatchSampler
import os 
from PIL import Image
import numpy as np
import torch 
import random
from torchvision import transforms


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
            self.image_paths.append(os.path.join(root_dir, row['img_name']+ '.png'))
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
    def __init__(self, df, root_dir, joint_transform = None, mask_use = False):
        self.df = df
        self.mask_use = mask_use
        self.joint_transform = joint_transform  # Add joint transform for image and mask
        self.image_paths = []
        self.mask_paths = []
        self.labels = []
        
        # Mapping for label encoding
        
        for idx, row in self.df.iterrows():
            self.image_paths.append(os.path.join(root_dir, 'Dataset', f'{row["ID"]}.png')) # for original
            
            if self.mask_use:
                self.mask_paths.append(os.path.join(root_dir, 'mask', f'{row["ID"]}.png')) #for med sam mask
            self.labels.append(row['label'])  
    
    def get_labels(self):
        return self.labels
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.mask_use:
            mask_path = self.mask_paths[idx]
            mask = Image.open(mask_path).convert('L')  # Keep mask as grayscale
        else:
            mask = Image.new('L', image.size) # 껍데기 mask

        
        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)
                
        return image, mask, torch.tensor(label)  # Convert label to tensor for model compatibility

class JointTransform:
    def __init__(self, resize=None, horizontal_flip=False, vertical_flip=False, rotation=False, interpolation=False, zoom = 0.1):
        self.resize = resize
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation = rotation
        self.interpolation = interpolation
        self.zoom = zoom 
        
    def __call__(self, image, mask):
        # Resize
        if self.resize:
            resize_transform = transforms.Resize(self.resize)
            image = resize_transform(image)
            mask = resize_transform(mask)
        
        # Random horizontal flip
        if self.horizontal_flip and random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        
        # Random vertical flip
        if self.vertical_flip and random.random() > 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
        
        # Random rotation
        if self.rotation:
            angle = random.randint(-self.rotation, self.rotation)
            image = transforms.functional.rotate(image, angle)
            mask = transforms.functional.rotate(mask, angle)
        
        if self.interpolation:
            image = transforms.functional.resize(image, self.resize, interpolation=self.interpolation)
            mask = transforms.functional.resize(mask, self.resize, interpolation=self.interpolation)
        
        # image는 2channel로 적용
        
        # To Tensor
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        
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