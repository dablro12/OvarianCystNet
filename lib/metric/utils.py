def load2json(path):
    import json
    with open(path, 'r') as f:
        return json.load(f)

def load_model_metrics(models_paths):
    model_metrics = {}
    for path in models_paths:
        model_name = path.split('/')[-1].split('_fold')[0]
        model_metrics[model_name] = load2json(path)
    return model_metrics


# dataframe 으로 변환시키기 
import pandas as pd
def metrics2df(metrics):
    df = pd.DataFrame()
    for model_name, fold_metrics in metrics.items():
        for fold, metric in fold_metrics.items():
            metric['model'] = model_name
            metric['fold'] = fold
            # DataFrame에 행을 추가할 때 concat을 사용
            df = pd.concat([df, pd.DataFrame([metric])], ignore_index=True)
            # 칼럼 인덱스 변경 1 : model, 2 : fold 
            df = df[['model', 'fold'] + [col for col in df.columns if col not in ['model', 'fold']]]
    return df


import os 
def model_fold_comparison_plot(save_metric_dir):
    """ 모델들의 각 메트릭별 비교 모델 그리기 """
    
    models_paths = [os.path.join(save_metric_dir, f) for f in os.listdir(save_metric_dir) if f.endswith('.json')]
    models_metrics = load_model_metrics(models_paths)
    df = metrics2df(models_metrics)
    
    return df

