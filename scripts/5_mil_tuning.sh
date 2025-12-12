# /bin/bash
# scripts/5_mil_tuning.sh
LR="5e-5"
# # Binary Class
# /opt/conda/bin/python /workspace/mil_baseline.py \
#     --data_root_dir "/workspace/pcos_dataset/Dataset" \
#     --label_path "/workspace/pcos_dataset/labels/기존_Dataset_info_binary.csv" \
#     --result_dir "/workspace/pcos_dataset/results/mil_task/binary/baseline_기존라벨_epoch100" \
#     --model_type "efficient" \
#     --num_epochs 100 \
#     --lr $LR \
#     --n_splits 5 \
#     --num_workers 16 \
#     --gpu_id 1

# /opt/conda/bin/python /workspace/mil_baseline.py \
#     --data_root_dir "/workspace/pcos_dataset/Dataset" \
#     --label_path "/workspace/pcos_dataset/labels/변경_Dataset_info_binary.csv" \
#     --result_dir "/workspace/pcos_dataset/results/mil_task/binary/baseline_변경라벨_epoch100" \
#     --model_type "transformer" \
#     --num_epochs 100 \
#     --lr $LR \
#     --n_splits 5 \
#     --num_workers 16 \
#     --gpu_id 1

# Multi Class
/opt/conda/bin/python /workspace/mil_baseline.py \
    --data_root_dir "/workspace/pcos_dataset/Dataset" \
    --label_path "/workspace/pcos_dataset/labels/기존_Dataset_info.csv" \
    --result_dir "/workspace/pcos_dataset/results/mil_task/multi/baseline_기존라벨" \
    --model_type "transformer" \
    --num_epochs 50 \
    --lr $LR \
    --n_splits 5 \
    --num_workers 16 \
    --gpu_id 1

# /opt/conda/bin/python /workspace/mil_baseline.py \
#     --data_root_dir "/workspace/pcos_dataset/Dataset" \
#     --label_path "/workspace/pcos_dataset/labels/변경_Dataset_info.csv" \
#     --result_dir "/workspace/pcos_dataset/results/mil_task/multi/baseline_변경라벨_epoch100" \
#     --model_type "transformer" \
#     --num_epochs 100 \
#     --lr $LR \
#     --n_splits 5 \
#     --num_workers 16 \
#     --gpu_id 1
