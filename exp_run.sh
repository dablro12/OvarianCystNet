#!/bin/bash

export NO_ALBUMENTATIONS_UPDATE=1
LR=0.000002

# /home/eiden/miniconda3/envs/medmamba/bin/python exp_train.py \
#     --model_name="medmamba" \
#     --model_type="CervicalUS" \
#     --lr=0.000005 \
#     --save_dir="/mnt/hdd/octc/PCOS_experiment/binary/checkpoint" \
#     --epochs=100 \
#     --wandb_use=True \
#     --patience=100 \
#     --run_name="augmixx2-polyloss" \

# /home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
#     --model_name="resnet" \
#     --model_type="resnet34" \
#     --lr=0.00001 \
#     --save_dir="/mnt/hdd/octc/PCOS_experiment/binary/checkpoint" \
#     --epochs=100 \
#     --wandb_use=True \
#     --patience=20 \
#     --run_name="augmixx2-polyloss" \


# /home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
#     --model_name="mobilenet" \
#     --model_type="l" \
#     --lr=0.00001 \
#     --save_dir="/mnt/hdd/octc/PCOS_experiment/binary/checkpoint" \
#     --epochs=100 \
#     --wandb_use=True \
#     --patience=20 \
#     --run_name="augmixx2-polyloss" \


# /home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
#     --model_name="convnext" \
#     --model_type="l" \
#     --lr=0.00001 \
#     --save_dir="/mnt/hdd/octc/PCOS_experiment/binary/checkpoint" \
#     --epochs=100 \
#     --wandb_use=True \
#     --patience=20 \
#     --run_name="augmixx2-polyloss" \


# /home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
#     --model_name="swin-transformer" \
#     --model_type="timm" \
#     --lr=0.00001 \
#     --save_dir="/mnt/hdd/octc/PCOS_experiment/binary/checkpoint" \
#     --epochs=100 \
#     --wandb_use=True \
#     --patience=20 \
#     --run_name="augmixx2-polyloss" \


# /home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
#     --model_name="vision-transformer" \
#     --model_type="timm" \
#     --lr=0.00001 \
#     --save_dir="/mnt/hdd/octc/PCOS_experiment/binary/checkpoint" \
#     --epochs=100 \
#     --wandb_use=True \
#     --patience=20 \
#     --run_name="augmixx2-polyloss01" \


# /home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
#     --model_name="maxvit" \
#     --model_type="default" \
#     --lr=0.00001 \
#     --save_dir="/mnt/hdd/octc/PCOS_experiment/binary/checkpoint" \
#     --epochs=100 \
#     --wandb_use=True \
#     --patience=20 \
#     --run_name="augmixx2-polyloss" \





####################<<<<<<<<<<<<<<<<------------------------------->>>>>>>>>############################
# /home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
#     --model_name="resnet" \
#     --model_type="resnet34" \
    # --lr=0.0001 \
#     --save_dir="/mnt/hdd/octc/PCOS_experiment/multi/checkpoint" \
#     --epochs=100 \
#     --wandb_use=True \
#     --patience=20 \
#     --run_name="auggancombinedx2v2-PolyLoss" \


# /home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
#     --model_name="mobilenet" \
#     --model_type="l" \
    # --lr=0.0001 \
#     --save_dir="/mnt/hdd/octc/PCOS_experiment/multi/checkpoint" \
#     --epochs=100 \
#     --wandb_use=True \
#     --patience=20 \
#     --run_name="auggancombinedx2v2-PolyLoss" \


# /home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
#     --model_name="convnext" \
#     --model_type="l" \
    # --lr=0.0001 \
#     --save_dir="/mnt/hdd/octc/PCOS_experiment/multi/checkpoint" \
#     --epochs=100 \
#     --wandb_use=True \
#     --patience=20 \
#     --run_name="auggancombinedx2v2-PolyLoss" \


# /home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
#     --model_name="swin-transformer" \
#     --model_type="timm" \
    # --lr=0.0001 \
#     --save_dir="/mnt/hdd/octc/PCOS_experiment/multi/checkpoint" \
#     --epochs=100 \
#     --wandb_use=True \
#     --patience=20 \
#     --run_name="auggancombinedx2v2-PolyLoss" \


# /home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
#     --model_name="vision-transformer" \
#     --model_type="timm" \
    # --lr=0.0001 \
#     --save_dir="/mnt/hdd/octc/PCOS_experiment/multi/checkpoint" \
#     --epochs=100 \
#     --wandb_use=True \
#     --patience=20 \
#     --run_name="auggancombinedx2v2-PolyLoss" \


# /home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
#     --model_name="maxvit" \
#     --model_type="default" \
    # --lr=0.0001 \
#     --save_dir="/mnt/hdd/octc/PCOS_experiment/multi/checkpoint" \
#     --epochs=100 \
#     --wandb_use=True \
#     --patience=20 \
#     --run_name="auggancombinedx2v2-PolyLoss" \


# /home/eiden/miniconda3/envs/USFM/bin/python exp_train.py \
#     --model_name="medmamba" \
#     --model_type="CervicalUS" \
    # --lr=0.0001 \
#     --save_dir="/mnt/hdd/octc/PCOS_experiment/multi/checkpoint" \
#     --epochs=100 \
#     --wandb_use=True \
#     --patience=20 \
#     --run_name="auggancombinedx2v2-PolyLoss" \

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<TESTING>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# /home/eiden/miniconda3/envs/biomedclip/bin/python exp_train.py \
#     --model_name="clip" \
#     --model_type="biomedclip-linearprob" \
#     --lr=$LR \
#     --save_dir="/mnt/hdd/octc/PCOS_experiment/binary/checkpoint" \
#     --epochs=100 \
#     --wandb_use=True \
#     --patience=100 \
#     --run_name="augmixx2-polyloss-lr0.000002gradtrue" \


# /home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
#     --model_name="swin-transformer" \
#     --model_type="timm" \
#     --lr=0.000002 \
#     --save_dir="/mnt/hdd/octc/PCOS_experiment/binary/checkpoint" \
#     --epochs=100 \
#     --wandb_use=True \
#     --patience=20 \
#     --run_name="augmixx2-polyloss-lr0.000002" \


# /home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
#     --model_name="vision-transformer" \
#     --model_type="timm" \
#     --lr=0.000002 \
#     --save_dir="/mnt/hdd/octc/PCOS_experiment/binary/checkpoint" \
#     --epochs=100 \
#     --wandb_use=True \
#     --patience=20 \
#     --run_name="augmixx2-polyloss-lr0.000002" \



# /home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
#     --model_name="swin-transformer" \
#     --model_type="timm" \
#     --lr=0.0000001 \
#     --save_dir="/mnt/hdd/octc/PCOS_experiment/binary/checkpoint" \
#     --epochs=100 \
#     --wandb_use=True \
#     --patience=20 \
#     --run_name="augmixx2-polyloss-lr0.0000001" \


# /home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
#     --model_name="vision-transformer" \
#     --model_type="timm" \
#     --lr=0.0000001 \
#     --save_dir="/mnt/hdd/octc/PCOS_experiment/binary/checkpoint" \
#     --epochs=100 \
#     --wandb_use=True \
#     --patience=20 \
#     --run_name="augmixx2-polyloss-lr0.0000001" \

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Stress test TESTING>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# 0 - 5
/home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
    --df_path="/mnt/hdd/octc/PCOS_Dataset/aug_train_csv/augcombined_num_4_train_split.csv" \
    --dataset_name="Dataset-combined-ACGAN" \
    --model_name="vision-transformer" \
    --model_type="timm" \
    --lr=$LR \
    --save_dir="/mnt/hdd/octc/PCOS_experiment/binary/checkpoint" \
    --epochs=100 \
    --wandb_use=True \
    --patience=50 \
    --run_name="poly-acgan-num_4" \

/home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
    --df_path="/mnt/hdd/octc/PCOS_Dataset/aug_train_csv/augcombined_num_4_train_split.csv" \
    --dataset_name="Dataset-combined-ACGAN" \
    --model_name="resnet" \
    --model_type="resnet34" \
    --lr=$LR \
    --save_dir="/mnt/hdd/octc/PCOS_experiment/binary/checkpoint" \
    --epochs=100 \
    --wandb_use=True \
    --patience=50 \
    --run_name="poly-acgan-num_4" \

/home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
    --df_path="/mnt/hdd/octc/PCOS_Dataset/aug_train_csv/augcombined_num_4_train_split.csv" \
    --dataset_name="Dataset-combined-ACGAN" \
    --model_name="mobilenet" \
    --model_type="l" \
    --lr=$LR \
    --save_dir="/mnt/hdd/octc/PCOS_experiment/binary/checkpoint" \
    --epochs=100 \
    --wandb_use=True \
    --patience=50 \
    --run_name="poly-acgan-num_4" \

/home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
    --df_path="/mnt/hdd/octc/PCOS_Dataset/aug_train_csv/augcombined_num_4_train_split.csv" \
    --dataset_name="Dataset-combined-ACGAN" \
    --model_name="convnext" \
    --model_type="l" \
    --lr=$LR \
    --save_dir="/mnt/hdd/octc/PCOS_experiment/binary/checkpoint" \
    --epochs=100 \
    --wandb_use=True \
    --patience=50 \
    --run_name="poly-acgan-num_4" \

/home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
    --df_path="/mnt/hdd/octc/PCOS_Dataset/aug_train_csv/augcombined_num_4_train_split.csv" \
    --dataset_name="Dataset-combined-ACGAN" \
    --model_name="maxvit" \
    --model_type="default" \
    --lr=$LR \
    --save_dir="/mnt/hdd/octc/PCOS_experiment/binary/checkpoint" \
    --epochs=100 \
    --wandb_use=True \
    --patience=50 \
    --run_name="poly-acgan-num_4" \

/home/eiden/miniconda3/envs/medmamba/bin/python exp_train.py \
    --df_path="/mnt/hdd/octc/PCOS_Dataset/aug_train_csv/augcombined_num_4_train_split.csv" \
    --dataset_name="Dataset-combined-ACGAN" \
    --model_name="medmamba" \
    --model_type="CervicalUS" \
    --lr=$LR \
    --save_dir="/mnt/hdd/octc/PCOS_experiment/binary/checkpoint" \
    --epochs=100 \
    --wandb_use=True \
    --patience=50 \
    --run_name="poly-acgan-num_4" \

