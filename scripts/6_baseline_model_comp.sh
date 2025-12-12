
## multi
GPU_DEVICE=1

#기존라벨
# /opt/conda/bin/python /workspace/hg_baseline.py \
#     --model_name "google/vit-base-patch16-224" \
#     --model_cache_dir "/workspace/pcos_dataset/models" \
#     --data_root_dir "/workspace/pcos_dataset/Dataset" \
#     --label_path "/workspace/pcos_dataset/labels/통합_Dataset_info_binary_nan.csv" \
#     --label_col_name "USG_Ontology" \
#     --result_root_dir "/workspace/pcos_dataset/results/label변경/binary/vit-base-patch16-224" \
#     --num_epochs 50 \
#     --learning_rate 6.373233701464723e-06  \
#     --batch_size 32 \
#     --n_splits 5 \
#     --gpu_id 1 \
#     --logging_backend "tensorboard" \
#     --wandb_project "pcos-ultrasound"

/opt/conda/bin/python /workspace/hg_baseline.py \
    --model_name "microsoft/resnet-50" \
    --model_cache_dir "/workspace/pcos_dataset/models" \
    --data_root_dir "/workspace/pcos_dataset/Dataset" \
    --label_path "/workspace/pcos_dataset/labels/통합_Dataset_info_binary.csv" \
    --label_col_name "USG_Ontology" \
    --result_root_dir "/workspace/pcos_dataset/results/label변경/binary/resnet-50" \
    --num_epochs 50 \
    --learning_rate 5e-5  \
    --batch_size 256 \
    --n_splits 5 \
    --gpu_id 1 \
    --logging_backend "tensorboard" \
    --wandb_project "pcos-ultrasound"


# /opt/conda/bin/python /workspace/hg_baseline.py \
#     --model_name "google/efficientnet-b0" \
#     --model_cache_dir "/workspace/pcos_dataset/models" \
#     --data_root_dir "/workspace/pcos_dataset/Dataset" \
#     --label_path "/workspace/pcos_dataset/labels/통합_Dataset_info_binary.csv" \
#     --label_col_name "USG_Ontology" \
#     --result_root_dir "/workspace/pcos_dataset/results/label변경/binary/efficientnet-b0" \
#     --num_epochs 50 \
#     --learning_rate 6.373233701464723e-06  \
#     --batch_size 32 \
#     --n_splits 5 \
#     --gpu_id 1 \
#     --logging_backend "tensorboard" \
#     --wandb_project "pcos-ultrasound"

# /opt/conda/bin/python /workspace/hg_baseline.py \
#     --model_name "facebook/convnext-tiny-224" \
#     --model_cache_dir "/workspace/pcos_dataset/models" \
#     --data_root_dir "/workspace/pcos_dataset/Dataset" \
#     --label_path "/workspace/pcos_dataset/labels/통합_Dataset_info_binary.csv" \
#     --label_col_name "USG_Ontology" \
#     --result_root_dir "/workspace/pcos_dataset/results/label변경/binary/convnext-tiny-224" \
#     --num_epochs 50 \
#     --learning_rate 6.373233701464723e-06  \
#     --batch_size 32 \
#     --n_splits 5 \
#     --gpu_id 1 \
#     --logging_backend "tensorboard" \
#     --wandb_project "pcos-ultrasound"

# /opt/conda/bin/python /workspace/hg_baseline.py \
#     --model_name "facebook/dinov2-base-imagenet1k-1-layer" \
#     --model_cache_dir "/workspace/pcos_dataset/models" \
#     --data_root_dir "/workspace/pcos_dataset/Dataset" \
#     --label_path "/workspace/pcos_dataset/labels/통합_Dataset_info_binary.csv" \
#     --label_col_name "USG_Ontology" \
#     --result_root_dir "/workspace/pcos_dataset/results/label변경/binary/dinov2-base-imagenet1k-1-layer" \
#     --num_epochs 50 \
#     --learning_rate 6.373233701464723e-06  \
#     --batch_size 32 \
#     --n_splits 5 \
#     --gpu_id 1 \
#     --logging_backend "tensorboard" \
#     --wandb_project "pcos-ultrasound"