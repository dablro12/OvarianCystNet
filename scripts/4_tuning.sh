#기존라벨
/opt/conda/bin/python /workspace/hg_baseline.py \
    --model_name "google/vit-base-patch16-224" \
    --model_cache_dir "/workspace/pcos_dataset/models" \
    --data_root_dir "/workspace/pcos_dataset/Dataset" \
    --label_path "/workspace/pcos_dataset/labels/기존_Dataset_info.csv" \
    --result_root_dir "/workspace/pcos_dataset/results/baseline_기존라벨" \
    --num_epochs 30 \
    --learning_rate 5e-4 \
    --batch_size 192 \
    --n_splits 5 \
    --gpu_id 1 \
    --logging_backend "tensorboard" \
    --wandb_project "pcos-ultrasound"

#변경라벨
/opt/conda/bin/python /workspace/hg_baseline.py \
    --model_name "google/vit-base-patch16-224" \
    --model_cache_dir "/workspace/pcos_dataset/models" \
    --data_root_dir "/workspace/pcos_dataset/Dataset" \
    --label_path "/workspace/pcos_dataset/labels/변경_Dataset_info.csv" \
    --result_root_dir "/workspace/pcos_dataset/results/baseline_변경라벨" \
    --num_epochs 30 \
    --learning_rate 5e-4 \
    --batch_size 192 \
    --n_splits 5 \
    --gpu_id 1 \
    --logging_backend "tensorboard" \
    --wandb_project "pcos-ultrasound"