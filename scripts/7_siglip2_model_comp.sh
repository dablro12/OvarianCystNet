
## Binary
#기존라벨

model_names=(
    "google/siglip2-so400m-patch16-384"
    "google/siglip2-base-patch32-256"
    "google/siglip2-base-patch16-384"
    "google/siglip2-so400m-patch14-384"
    "google/siglip2-large-patch16-384"
)

for model_name in "${model_names[@]}"; do
    # 모델명에서 마지막 '-' 이후의 값을 추출하여 data_resize_size로 사용
    /opt/conda/bin/python /workspace/siglip2_trainer.py \
        --model_name "$model_name" \
        --model_cache_dir "/workspace/pcos_dataset/models" \
        --data_root_dir "/workspace/pcos_dataset/Dataset" \
        --label_path "/workspace/pcos_dataset/labels/통합_Dataset_info.csv" \
        --label_col_name "USG_Ontology" \
        --result_root_dir "/workspace/pcos_dataset/results/siglip2_baseline/multi/${model_name//\//_}" \
        --num_epochs 50 \
        --learning_rate 6.373233701464723e-06  \
        --batch_size 16 \
        --n_splits 5 \
        --gpu_id 1 \
        --logging_backend "tensorboard" \
        --wandb_project "pcos-ultrasound"

    /opt/conda/bin/python /workspace/siglip2_trainer.py \
        --model_name "$model_name" \
        --model_cache_dir "/workspace/pcos_dataset/models" \
        --data_root_dir "/workspace/pcos_dataset/Dataset" \
        --label_path "/workspace/pcos_dataset/labels/통합_Dataset_info_binary.csv" \
        --label_col_name "USG_Ontology" \
        --result_root_dir "/workspace/pcos_dataset/results/siglip2_baseline/binary/${model_name//\//_}" \
        --num_epochs 50 \
        --learning_rate 6.373233701464723e-06  \
        --batch_size 16 \
        --n_splits 5 \
        --gpu_id 1 \
        --logging_backend "tensorboard" \
        --wandb_project "pcos-ultrasound"
done