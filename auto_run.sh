#!/bin/bash

# 명령어가 실패하면 즉시 종료
set -e

# .env 파일 경로
ENV_FILE=".env"

# .env 파일이 존재하는지 확인
if [ -f "$ENV_FILE" ]; then
    echo ".env 파일에서 환경 변수를 로드합니다."
    # 주석과 빈 줄을 제외하고 .env 파일의 변수를 가져옵니다.
    export $(grep -v '^#' $ENV_FILE | xargs)
else
    echo "오류: $ENV_FILE 파일을 찾을 수 없습니다!"
    exit 1
fi

# 가상 환경 활성화 (필요한 경우 경로를 설정하고 주석 해제)
# source /path/to/venv/bin/activate

# 로그 디렉토리가 없으면 생성
LOG_DIR="log"
mkdir -p "$LOG_DIR"

# VER와 MASK_USE의 조합 정의
declare -a VERSIONS=("origin" "inpaint" "mask")
# declare -a VERSIONS=("originwithaugment" "inpaintnwithaugment" "masknwithaugment")

# 모델과 타입의 조합 정의
declare -a MASK_USES=("no" "yes" "yes")
declare -a BACKBONE_MODELS=("resnet" "mobilenet" "efficient" "convnext" "swin-transformer" "vision-transformer" "maxvit")

declare -A MODEL_TYPES
MODEL_TYPES["resnet"]="resnet34"
MODEL_TYPES["mobilenet"]="l"
MODEL_TYPES["efficient"]="l"
MODEL_TYPES["convnext"]="l"
MODEL_TYPES["swin-transformer"]="default"
MODEL_TYPES["vision-transformer"]="l_16"
MODEL_TYPES["maxvit"]="default"
# 필요한 경우 다른 모델과 타입 추가

# VER와 MASK_USE의 조합을 순회
for idx in "${!VERSIONS[@]}"; do
    VER="${VERSIONS[$idx]}"
    MASK_USE="${MASK_USES[$idx]}"

    # 모델과 타입을 순회
    for BACKBONE_MODEL in "${BACKBONE_MODELS[@]}"; do
        # 모델에 맞는 타입 가져오기
        MODEL_TYPE="${MODEL_TYPES[$BACKBONE_MODEL]}"
        
        # 환경 변수 설정
        export VER
        export MASK_USE
        export BACKBONE_MODEL
        export MODEL_TYPE

        # 로그 파일 정의
        LOG_FILE="$LOG_DIR/${BACKBONE_MODEL}_${MODEL_TYPE}_${VER}.txt"

        # 훈련 스크립트 실행
        nohup /home/eiden/miniconda3/envs/cv/bin/python3 "$PYTHON_SCRIPT" \
            --wandb_use "$WANDB_USE" \
            --wandb_project "$WANDB_PROJECT" \
            --data_dir "$DATA_DIR" \
            --csv_path "$CSV_PATH" \
            --version "$VER" \
            --mask_use "$MASK_USE" \
            --fold_num "$FOLD_NUM" \
            --train_batch_size "$TRAIN_BATCH_SIZE" \
            --valid_batch_size "$VALID_BATCH_SIZE" \
            --lr "$LR" \
            --epochs "$EPOCHS" \
            --backbone_model "$BACKBONE_MODEL" \
            --type "$MODEL_TYPE" \
            --save_dir "$SAVE_DIR" \
            --seed "$RANDOM_SEED" \
            --outlayer_num "$OUTLAYER_NUM" \
            > "$LOG_FILE" 2>&1 &
        echo "모델: $BACKBONE_MODEL, 타입: $MODEL_TYPE, 버전: $VER, 마스크 사용: $MASK_USE 에 대한 훈련을 시작했습니다. 로그는 $LOG_FILE에 저장됩니다."

        # 시스템 리소스를 고려하여 각 작업이 완료될 때까지 기다림
        wait
    done
done