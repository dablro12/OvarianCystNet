#!/bin/bash

export NO_ALBUMENTATIONS_UPDATE=1

# /home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
#     --model_name="resnet" \
#     --model_type="resnet34" \
#     --save_dir="/mnt/hdd/octc/PCOS_experiment/binary/checkpoint" \
#     --epochs=50 \
#     --wandb_use=True \
#     --patience=10 \
#     --run_name="auggancombinedx2-lr0.0001" \


# /home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
#     --model_name="mobilenet" \
#     --model_type="l" \
#     --save_dir="/mnt/hdd/octc/PCOS_experiment/binary/checkpoint" \
#     --epochs=50 \
#     --wandb_use=True \
#     --patience=10 \
#     --run_name="auggancombinedx2-lr0.0001" \


# /home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
#     --model_name="convnext" \
#     --model_type="l" \
#     --save_dir="/mnt/hdd/octc/PCOS_experiment/binary/checkpoint" \
#     --epochs=50 \
#     --wandb_use=True \
#     --patience=10 \
#     --run_name="auggancombinedx2-lr0.0001" \


# /home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
#     --model_name="swin-transformer" \
#     --model_type="default" \
#     --save_dir="/mnt/hdd/octc/PCOS_experiment/binary/checkpoint" \
#     --epochs=50 \
#     --wandb_use=True \
#     --patience=10 \
#     --run_name="auggancombinedx2-lr0.0001" \


# /home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
#     --model_name="vision-transformer" \
#     --model_type="l_16" \
#     --save_dir="/mnt/hdd/octc/PCOS_experiment/binary/checkpoint" \
#     --epochs=50 \
#     --wandb_use=True \
#     --patience=10 \
#     --run_name="auggancombinedx2-lr0.0001" \


# /home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
#     --model_name="maxvit" \
#     --model_type="default" \
#     --save_dir="/mnt/hdd/octc/PCOS_experiment/binary/checkpoint" \
#     --epochs=50 \
#     --wandb_use=True \
#     --patience=10 \
#     --run_name="auggancombinedx2-lr0.0001" \


/home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
    --model_name="resnet" \
    --model_type="resnet34" \
    --save_dir="/mnt/hdd/octc/PCOS_experiment/multi/checkpoint" \
    --epochs=100 \
    --wandb_use=True \
    --patience=20 \
    --run_name="auggancombinedx2v2-PolyLoss" \


/home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
    --model_name="mobilenet" \
    --model_type="l" \
    --save_dir="/mnt/hdd/octc/PCOS_experiment/multi/checkpoint" \
    --epochs=100 \
    --wandb_use=True \
    --patience=20 \
    --run_name="auggancombinedx2v2-PolyLoss" \


/home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
    --model_name="convnext" \
    --model_type="l" \
    --save_dir="/mnt/hdd/octc/PCOS_experiment/multi/checkpoint" \
    --epochs=100 \
    --wandb_use=True \
    --patience=20 \
    --run_name="auggancombinedx2v2-PolyLoss" \


/home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
    --model_name="swin-transformer" \
    --model_type="default" \
    --save_dir="/mnt/hdd/octc/PCOS_experiment/multi/checkpoint" \
    --epochs=100 \
    --wandb_use=True \
    --patience=20 \
    --run_name="auggancombinedx2v2-PolyLoss" \


/home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
    --model_name="vision-transformer" \
    --model_type="l_16" \
    --save_dir="/mnt/hdd/octc/PCOS_experiment/multi/checkpoint" \
    --epochs=100 \
    --wandb_use=True \
    --patience=20 \
    --run_name="auggancombinedx2v2-PolyLoss" \


/home/eiden/miniconda3/envs/cv/bin/python exp_train.py \
    --model_name="maxvit" \
    --model_type="default" \
    --save_dir="/mnt/hdd/octc/PCOS_experiment/multi/checkpoint" \
    --epochs=100 \
    --wandb_use=True \
    --patience=20 \
    --run_name="auggancombinedx2v2-PolyLoss" \

