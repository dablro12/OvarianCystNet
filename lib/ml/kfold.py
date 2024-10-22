from sklearn.model_selection import GroupKFold, KFold
import pandas as pd 
def k_fold_split(csv_path, random_seed =42 ):
    df = pd.read_csv(csv_path)
    groups = df['ID']  # Assuming 'pid' is the column for patient IDs
    y = df['label']     # Assuming 'label' is the target variable

    # folder = GroupKFold(n_splits=5)
    folder = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    folds = []

    # for train_idx, val_idx in folder.split(df, y, groups):
    for train_idx, val_idx in folder.split(df, y, groups):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        fold = {'train': train_df, 'val': val_df}
        folds.append(fold)
    return folds
