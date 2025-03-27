from trainer import PCOS_trainer
import torch
import os
import datetime 
import optuna
from optuna.samplers import GridSampler


def objective(trial):
    """
    Optuna가 반복적으로 호출하는 objective 함수.
    - trial: Optuna 내부에서 현재 시도하는 하이퍼파라미터를 담고 있는 객체
    """

    # 하이퍼파라미터 최적화
    label_smoothing = trial.suggest_uniform('label_smoothing', 0.0, 0.3)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    # label_smoothing = 0.19215890194834306
    # lr = 0.00018223330318900866
    
    patience = 18
    # scheduler 관련 unified parameters (모든 스케줄러에 동일하게 적용)
    scheduler_type = 'cosine'            # 스케줄러 종류: 'cosine' / 'plateau' / 'step' / 'multistep' / 'cyclic' / 'warmrestarts'
    scheduler_factor = 0.1               # ReduceLROnPlateau에서 사용 (학습률 감소 비율)
    scheduler_patience = patience // 2   # ReduceLROnPlateau에서 사용 (개선 없는 에폭 수)
    scheduler_T_max = 50 - 5             # CosineAnnealingLR: (epoch - warmup_epochs)
    scheduler_eta_min = 1e-6             # CosineAnnealingLR에서 사용 (최소 학습률)
    scheduler_step_size = 10             # StepLR에서 사용 (에폭 단위 감소 주기)
    scheduler_gamma = 0.5                # StepLR, MultiStepLR, CyclicLR 등에서 사용 (감소 비율)
    scheduler_milestones = [10, 30, 50]    # MultiStepLR에서 사용 (감소 시점)
    scheduler_max_lr_scale = 10.0        # CyclicLR에서 사용 (최대 lr 배수)
    scheduler_step_size_up = 5           # CyclicLR: 상승 단계 에폭 수
    scheduler_step_size_down = 5         # CyclicLR: 하강 단계 에폭 수
    scheduler_T0 = 10                    # CosineAnnealingWarmRestarts에서 사용 (첫 주기 길이)
    scheduler_T_mult = 1                 # CosineAnnealingWarmRestarts에서 사용 (이후 주기 배수)
    scheduler_eta_min_scale = 0.1        # CosineAnnealingWarmRestarts에서 사용 (최소 lr 비율)


    # Args 생성 및 초기화
    class Args:
        pass
    args = Args()
    
    #%% [WanDB]
    args.use_wandb = True
    args.project_name = "PCOS_Classification"
    args.experiment_name = "Model"

    #%% [Logging]
    args.exp_date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    #%% [Args]
    args.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    args.datasheet_path = os.getenv('DATASHEET_PATH')
    args.data_dir = os.getenv('DATA_DIR')
    
    #%% [Model]
    args.model_name = 'resnet'
    args.model_version = '18'  # 모델 버전 수정 (예: '18')
    
    #%% [Hyperparameters]
    args.loss_name = 'polyl1ce'  # None / 'polyl1ce' / 'poly1focal'
    args.epoch = 50              # 총 에폭 수
    args.warmup_epochs = 5        # Warmup 에폭 수
    args.patience = patience
    args.loss_label_smoothing = label_smoothing
    args.lr = lr


    #%% [Scheduler] Unified Parameters 설정
    args.scheduler_type = scheduler_type
    args.scheduler_factor = scheduler_factor
    args.scheduler_patience = scheduler_patience
    args.scheduler_T_max = scheduler_T_max
    args.scheduler_eta_min = scheduler_eta_min
    args.scheduler_step_size = scheduler_step_size
    args.scheduler_gamma = scheduler_gamma
    args.scheduler_milestones = scheduler_milestones
    args.scheduler_max_lr_scale = scheduler_max_lr_scale
    args.scheduler_step_size_up = scheduler_step_size_up
    args.scheduler_step_size_down = scheduler_step_size_down
    args.scheduler_T0 = scheduler_T0
    args.scheduler_T_mult = scheduler_T_mult
    args.scheduler_eta_min_scale = scheduler_eta_min_scale

    #%% [Data]
    # 여기서는 다중 클래스 실험으로 설정
    args.binary_use = False
    args.sampler_name = None  # 또는 'balanced' / 'weighted'
    args.use_kfold = True      # 단일 분할로 실험 (optuna에서는 빠른 실험을 위해)
    args.k_fold = 5
    
    #%% [Model]
    args.model_name = 'resnet'
    args.model_version = '18'
    
    #%% [Experiment Name]
    args.experiment_name = f"KFold_resnet_18_params_search"
    
    # Trainer 생성 및 학습 실행
    trainer = PCOS_trainer(args)
    # best_val_auc = trainer.fit()  # 여기서는 k_fold_fit 대신 fit() 사용 (빠른 실험을 위해)
    best_val_auc = trainer.k_fold_fit()  # 여기서는 k_fold_fit 대신 fit() 사용 (빠른 실험을 위해)

    # Return 값: 최적의 검증 AUC (Optuna는 이 값을 최대화하려고 함)
    return best_val_auc

def run_optuna_grid_search():
    # 그리드 서치할 하이퍼파라미터 공간 정의
    search_space = {
        'scheduler_type': ['cosine', 'StepLR', 'MultiStepLR', 'CyclicLR', 'CosineAnnealingWarmRestarts']
    }

    # GridSampler를 생성합니다.
    sampler = GridSampler(search_space)

    # Study 생성 (목표는 최대화)
    study = sampler.create_study(direction='maximize')

    # grid search에서는 n_trials를 하이퍼파라미터 조합 개수로 지정합니다.
    study.optimize(objective, n_trials=len(search_space['scheduler_type']), show_progress_bar= True)

    print("\n===== [Optuna Grid Search 결과] =====")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (Val AUC): {study.best_trial.value:.4f}")
    print("Best hyperparams:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")
        
def run_optuna_search():
    """
    #### Optuna Hyperparameter Tuning : 수치형
    Optuna Study를 생성/실행하고 최적 파라미터를 출력합니다.
    # 만약 무작위 탐색을 원한다면 아래 함수를 사용하세요.
    # 단, 이 경우 5번의 시도 중 일부 옵션이 중복 선택될 수 있으므로 모든 스케줄러 옵션이 반드시 테스트되지 않을 수 있습니다.
    """
    # direction='maximize' → best_val_auc가 높을수록 좋다고 판단합니다.
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5, show_progress_bar= True)  # n_trials=5: 총 5번 랜덤으로 시도 (실험 수 늘리면 더 많은 탐색 가능)

    print("\n===== [Optuna Search 결과] =====")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (Val AUC): {study.best_trial.value:.4f}")
    print("Best hyperparams:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")
if __name__ == '__main__':
    run_optuna_search() #<-- Parameter Search 실행
    # run_optuna_grid_search()
