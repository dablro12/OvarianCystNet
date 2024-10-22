import pandas as pd

def validate_pid_distributions(folds, train_df, test_df):
    """
    데이터 분할 후, 동일한 pid가 여러 세트에 포함되지 않았는지 검증합니다.
    
    Parameters:
    - folds: K-Fold 분할 결과 리스트 (각 요소는 {'train': DataFrame, 'val': DataFrame} 형식)
    - train_df: 전체 Train 세트 DataFrame
    - test_df: Test 세트 DataFrame
    """
    
    # 1. Train과 Test 세트 간의 pid 중복 확인
    train_pids = set(train_df['pid'])
    test_pids = set(test_df['pid'])
    
    overlap_pids = train_pids.intersection(test_pids)
    if overlap_pids:
        print("Error: Train과 Test 세트에 중복된 pid가 있습니다.")
        print(f"중복된 pid 개수: {len(overlap_pids)}")
        print(f"중복된 pid 목록: {overlap_pids}")
    else:
        print("확인 완료: Train과 Test 세트에 중복된 pid가 없습니다.")
    
    # 2. 각 폴드 내 Train과 Val 세트 간의 pid 중복 확인
    for i, fold in enumerate(folds, 1):
        fold_train_pids = set(fold['train']['pid'])
        fold_val_pids = set(fold['val']['pid'])
        
        overlap = fold_train_pids.intersection(fold_val_pids)
        if overlap:
            print(f"Error: 폴드 {i}의 Train과 Val 세트에 중복된 pid가 있습니다.")
            print(f"중복된 pid 개수: {len(overlap)}")
            print(f"중복된 pid 목록: {overlap}")
        else:
            print(f"확인 완료: 폴드 {i}의 Train과 Val 세트에 중복된 pid가 없습니다.")
    
    # 3. 모든 폴드의 Val 세트 간 pid 중복 확인
    val_pids_all_folds = [set(fold['val']['pid']) for fold in folds]
    all_val_pids = set()
    overlap_val_pids = set()
    
    for i, val_pids in enumerate(val_pids_all_folds, 1):
        overlapping = all_val_pids.intersection(val_pids)
        if overlapping:
            overlap_val_pids.update(overlapping)
            print(f"Error: 폴드 {i}의 Val 세트에 이미 다른 폴드의 Val 세트에 포함된 pid가 있습니다.")
            print(f"중복된 pid 개수: {len(overlapping)}")
            print(f"중복된 pid 목록: {overlapping}")
        all_val_pids.update(val_pids)
    
    if not overlap_val_pids:
        print("확인 완료: 모든 폴드의 Val 세트에 중복된 pid가 없습니다.")
    else:
        print("Error: 일부 Val 세트에 중복된 pid가 있습니다.")
        print(f"중복된 pid 개수: {len(overlap_val_pids)}")
        print(f"중복된 pid 목록: {overlap_val_pids}")

def print_label_distribution(folds, train_df, test_df):
    """
    데이터 분할 후, 각 세트의 레이블 분포를 출력합니다.
    
    Parameters:
    - folds: K-Fold 분할 결과 리스트 (각 요소는 {'train': DataFrame, 'val': DataFrame} 형식)
    - train_df: 전체 Train 세트 DataFrame
    - test_df: Test 세트 DataFrame
    """
    print("\n>>> 전체 Train 세트 레이블 분포 >>>")
    print(f"# Train Set 수 : {len(train_df)}")
    print(train_df['pid'].value_counts(normalize=True))
    
    print("\n>>> 전체 Test 세트 레이블 분포 >>>")
    print(f"# Test Set 수 : {len(test_df)}")
    print(test_df['pid'].value_counts(normalize=True))
    
    # for i, fold in enumerate(folds, 1):
    #     print(f"\n>>> 폴드 {i} - Train 세트 레이블 분포 >>>")
    #     print(fold['train']['pid_label'].value_counts(normalize=True))
        
    #     print(f">>> 폴드 {i} - Val 세트 레이블 분포 >>>")
    #     print(fold['val']['pid_label'].value_counts(normalize=True))

