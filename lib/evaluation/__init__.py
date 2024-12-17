# checkpoint파일에서 가장 큰 epoch인 파일 찾기 ex) best_model_fold_1_epoch_8.pth
import torch 
import sys, os
sys.path.append('../../')
from model.loader import model_Loader
def max_epoch_selector(checkpoints:list):
    max_epoch, max_epoch_idx = 0, 0
    for idx, checkpoint in enumerate(checkpoints):
        epoch = checkpoint.split('.pth')[0].split('_')[-1]
        if int(epoch) > max_epoch:
            max_epoch, max_epoch_idx = int(epoch), idx
    return max_epoch_idx

def bestmodel_selector(outlayer_num, checkpoint_root_dir:str, model_cards:list, idx:int, device = 'cpu'): 
    model_card = model_cards[idx]
    checkpoints = sorted(os.listdir(os.path.join(checkpoint_root_dir, model_card)))
    max_epoch = checkpoints[max_epoch_selector(checkpoints)]
    
    checkpoint_path = os.path.join(checkpoint_root_dir, model_card, max_epoch)
    
    model_name = model_card.split('_')[0]
    if 'vision-transformer_l_16' in model_card:
        model_type = model_card.split('_')[1] + '_' + model_card.split('_')[2]
    else:        
        model_type = model_card.split('_')[1]
    fold_num = model_card.split('_')[-1].split('-')[0]
    if 'ft' in model_card:
        # convnext_l_fold_1-mask-ft-medsam1
        versions = model_card.split('-')
        # versions를 - 로 합치기 => ex) version = mask-ft-medsam1
        version = '-'.join(versions[1:])
        #version: transformer_default_fold_5-mask-ft-medsam1 -> mask-ft-medsam1
        if model_name == 'swin-transformer':
            version = '-'.join(versions[2:])
    else:
        version = model_card.split('_')[-1].split('-')[1]
    
    if '-' in model_type:
        model_type = model_type.replace('-', '_') 
    
    checkpoint_model = model_Loader(model_name, outlayer_num = outlayer_num, type = model_type).to(device)
    model_weights = torch.load(checkpoint_path)['model_state_dict']
    checkpoint_model.load_state_dict(model_weights)
    
    print(f"Checkpont Install Complete!! checkpoint_model: {model_name} fold : {fold_num} version: {version}")
    return checkpoint_model, model_name, version, fold_num
