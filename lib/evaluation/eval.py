import sys, os 
sys.path.append('../../')
from lib.evaluation import * 
# testloader building 
from lib.dataset import Custom_pcos_dataset, JointTransform 
from lib.metric.metrics import multi_classify_metrics, binary_classify_metrics
import pickle 
import pandas as pd
import gc
from tqdm import tqdm
import torch 
from torch.utils.data import DataLoader

def memory_release(model):
    gc.collect()
    torch.cuda.empty_cache()
    del model

def dataloader_builder(test_csv_path:str, root_dir:str, class_num:int, input_res = (3, 224, 224), bs_size = 40, mask_use = False):
    test_augment_list = JointTransform(
        resize=(input_res[1], input_res[2]),
        horizontal_flip=False,
        vertical_flip=False,
        rotation=0,
        interpolation=False,
    )
    test_dataset = Custom_pcos_dataset(
        df = pd.read_csv(test_csv_path),
        root_dir = root_dir,
        joint_transform = test_augment_list,
        mask_use = mask_use,
        class_num = class_num,
    )
    test_loader = DataLoader(dataset = test_dataset, batch_size = bs_size, shuffle = False, num_workers=16)

    return test_loader

def test_inference(model, test_loader, mask_use, device, outlayer_num):
    all_labels, all_preds, all_probs = [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, masks, labels in test_loader:
            inputs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            if mask_use:
                inputs = inputs[:, :2, :, :]
                masks = masks.to(device, non_blocking=True)
                inputs = torch.cat([inputs, masks], dim=1)
            
            outputs = model(inputs)
            if outlayer_num > 1:
                # Multi-class classification
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                labels = labels.long()
            else:
                probs = torch.sigmoid(outputs).squeeze()
                predicted = (probs >= 0.5).int()
            
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            all_preds.extend(predicted.detach().cpu().numpy())
    
    print("Inference Complete!!")
    print("Computing Metrics...")
            
    # After the loop, compute metrics based on the classification type
    if outlayer_num > 1:  # Multi-class classification
        metrics = multi_classify_metrics(all_labels, all_preds, all_probs)
    else:  # Binary classification
        metrics = binary_classify_metrics(all_labels, all_preds, all_probs, test_on=True)
    
    # Add data for ROC curve and confusion matrix plotting
    metrics['all_labels'] = all_labels
    metrics['all_preds'] = all_preds
    metrics['all_probs'] = all_probs

    return metrics



def evaluation(outlayer_num, model_cards, checkpoint_root_dir, mask_use, device, test_loader, save_metric_dir):
    for idx, model_card in enumerate(tqdm(model_cards, desc="Model Evaluation Progress")):
        # Model Setup
        model, model_name, version, fold_num = bestmodel_selector(outlayer_num, checkpoint_root_dir, model_cards, idx, device)
        # Inference
        # if os.path.join(save_metric_dir, f'{model_name}_{version}_{fold_num}.pkl') not in os.listdir(save_metric_dir):
        metrics = test_inference(model, test_loader, mask_use, device, outlayer_num)
        # memory 초기화
        memory_release(model)
        
        try:
            with open(os.path.join(save_metric_dir, f'{model_name}_{version}_{fold_num}.pkl'), 'wb') as f:
                pickle.dump(metrics, f)
            print("Metrics Save Complete!!", f'{model_name}_{version}_{fold_num}.pkl')
        except MemoryError:
            print(f"Memory Error: {model_name}_{version}_{fold_num}")
