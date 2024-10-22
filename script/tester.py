#!/usr/bin/env python
import sys, os
sys.path.append('../')
import warnings
import torch 
from lib.seed import set_seed
import pickle 
sys.path.append('../')
# checkpoint파일에서 가장 큰 epoch인 파일 찾기 ex) best_model_fold_1_epoch_8.pth
import torch 
# testloader building 
from lib.dataset import Custom_bus_dataset, JointTransform 
from torch.utils.data import DataLoader
from lib.metric.metrics import multi_classify_metrics, binary_classify_metrics
import pandas as pd
import gc
from model.loader import model_Loader

# 특정 경고 메시지 무시
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning)
def max_epoch_selector(checkpoints:list):
    max_epoch, max_epoch_idx = 0, 0
    for idx, checkpoint in enumerate(checkpoints):
        epoch = checkpoint.split('.pth')[0].split('_')[-1]
        if int(epoch) > max_epoch:
            max_epoch, max_epoch_idx = int(epoch), idx
    return max_epoch_idx

def bestmodel_selector(checkpoint_root_dir:str, model_cards:list, idx:int, device = 'cpu'): 
    model_card = model_cards[idx]
    checkpoints = sorted(os.listdir(os.path.join(checkpoint_root_dir, model_card)))
    max_epoch = checkpoints[max_epoch_selector(checkpoints)]
    
    checkpoint_path = os.path.join(checkpoint_root_dir, model_card, max_epoch)
    
    model_name = model_card.split('_')[0]
    model_type = model_card.split('_')[1]
    fold_num = model_card.split('_')[-1].split('-')[0]
    version = model_card.split('_')[-1].split('-')[1]
    
    if '-' in model_type:
        model_type = model_type.replace('-', '_') 
    
    checkpoint_model = model_Loader(model_name, outlayer_num = 1, type = model_type).to(device)
    model_weights = torch.load(checkpoint_path)['model_state_dict']
    checkpoint_model.load_state_dict(model_weights) 
    print(f"Checkpont Install Complete!! checkpoint_model: {model_name} fold : {fold_num} version: {version}")
    return checkpoint_model, model_name, version, fold_num


def memory_release(model):
    gc.collect()
    torch.cuda.empty_cache()
    del model

def dataloader_builder(test_csv_path:str, root_dir:str, input_res = (3, 224, 224), bs_size = 40):
    test_augment_list = JointTransform(
        resize=(input_res[1], input_res[2]),
        horizontal_flip=False,
        vertical_flip=False,
        rotation=0,
        interpolation=False,
    )
    test_dataset = Custom_bus_dataset(
        df = pd.read_csv(test_csv_path),
        root_dir = root_dir,
        joint_transform = test_augment_list 
    )
    test_loader = DataLoader(dataset = test_dataset, batch_size = bs_size, shuffle = False, num_workers=16)

    return test_loader

def test_inference(model, test_loader, version, device):
    all_labels, all_preds, all_probs = [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, masks, labels in test_loader:
            labels = labels.to(device, non_blocking=True).float()
            if version =='mask':
                inputs = imgs.to(device, non_blocking=True)[:, :2, :, :] # for image(2) + mask(1)
                masks = masks.to(device, non_blocking=True)
                outputs = model(torch.concat([inputs, masks], dim=1)) # for image + mask 
            else:
                inputs = imgs.to(device, non_blocking=True)
                outputs = model(inputs) # for image + mask 
            probs = torch.sigmoid(outputs)
            predicted = torch.round(probs)
            
            all_labels.extend(labels.detach().cpu().numpy())
            all_preds.extend(predicted.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
        
    metrics = binary_classify_metrics(all_labels, all_preds, all_probs, test_on = True)
    return metrics 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # def main():
    set_seed(42)
    test_csv_path = '/mnt/hdd/octc/BACKUP/BreastUS/experiment/test_df.csv'
    root_dir = '/mnt/hdd/octc/BACKUP/BreastUS'
    checkpoint_root_dir = '/mnt/hdd/octc/BACKUP/BreastUS/experiment/checkpoint'
    save_metric_dir = '/mnt/hdd/octc/BACKUP/BreastUS/experiment/metrics'
    fold_num = 5
    seed = 42
    outlayer_num = 1


    # DataLoad
    test_df = pd.read_csv(test_csv_path)
    test_loader = dataloader_builder(test_csv_path, root_dir, input_res = (3, 224, 224), bs_size = 24)
    model_cards = sorted(os.listdir(checkpoint_root_dir))
    for idx, model_card in enumerate(model_cards):
        # Model Setup
        model, model_name, version, fold_num = bestmodel_selector(checkpoint_root_dir, model_cards, idx, device)

        # Inference
        # if os.path.join(save_metric_dir, f'{model_name}_{version}_{fold_num}.pkl') not in os.listdir(save_metric_dir):
        metrics = test_inference(model, test_loader, version, device)
        # memory 초기화
        memory_release(model)
        
        try:
            with open(os.path.join(save_metric_dir, f'{model_name}_{version}_{fold_num}.pkl'), 'wb') as f:
                pickle.dump(metrics, f)
            print("Metrics Save Complete!!", f'{model_name}_{version}_{fold_num}.pkl')
        except MemoryError:
            print(f"Memory Error: {model_name}_{version}_{fold_num}")
