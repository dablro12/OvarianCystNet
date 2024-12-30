#!/usr/bin/env python
import argparse

# multi_exp_classification 모듈 import
from script.exp_train import multi_exp_classification, binary_exp_classification

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required= True, type=str, help="모델 이름")
    parser.add_argument("--model_type", required= True, type=str, help="모델 타입")
    parser.add_argument("--save_dir", required= True, type=str, help="모델 저장 디렉토리")
    parser.add_argument("--epochs", required= True, type=int, default=100, help="학습 에폭 수")
    parser.add_argument("--patience", required= True, type=int, default=20, help="Early stopping patience")
    parser.add_argument("--wandb_use", required= True, type=bool, default=False, help="WandB 로깅 사용 여부")
    parser.add_argument("--run_name", required= True, type=str, default="tester", help="WandB run name")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # 인스턴스 생성
    # trainer = multi_exp_classification(args=args)
    trainer = binary_exp_classification(args=args)
    # 필요한 속성 주입
    trainer.run_name = args.run_name  # wandb run 이름
    trainer.wandb_use = args.wandb_use
    
    # 학습 실행
    trainer.fit(epochs=args.epochs)

if __name__ == "__main__":
    main()
