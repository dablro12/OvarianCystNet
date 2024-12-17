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

# 로그 디렉토리가 없으면 생성
LOG_DIR="log"
mkdir -p "$LOG_DIR"

# VER와 MASK_USE의 조합 정의 (한 번에 선언)
declare -a VERSIONS=("origin" "origin-augment" "inpaint" "inpaint-augment" "mask" "mask-augment")
declare -a MASK_USES=("no" "no" "no" "no" "yes" "yes")

# 모델과 타입의 조합 정의
declare -a BACKBONE_MODELS=("resnet" "mobilenet" "efficient" "convnext" "swin-transformer" "vision-transformer" "maxvit")

declare -A MODEL_TYPES
MODEL_TYPES["resnet"]="resnet34"
MODEL_TYPES["mobilenet"]="l"
MODEL_TYPES["efficient"]="l"
MODEL_TYPES["convnext"]="l"
MODEL_TYPES["swin-transformer"]="default"
MODEL_TYPES["vision-transformer"]="l_16"
MODEL_TYPES["maxvit"]="default"

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

        #출력
        echo $LOG_FILE
        echo $MASK_USE
    done
done
