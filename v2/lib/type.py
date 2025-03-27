import numpy as np 
def convert_np_number_to_python(obj):
    """ 재귀적으로 순회하며 np.float32, np.int64 등 NumPy 타입을 파이썬 기본 타입으로 바꿔준다 """
    if isinstance(obj, dict):
        return {k: convert_np_number_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_number_to_python(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_np_number_to_python(x) for x in obj)
    else:
        # float32, float64, int64 등등을 파이썬 기본형으로 변환
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        # 그 외에는 그대로 반환
        return obj
