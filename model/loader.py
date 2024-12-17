# model/loader.py
import importlib

MODEL_LIST = [
    "alexnet", "convnext", "dynamic-vit", "inception", "mobilenet", "resnet", "vgg",
    "densenet", "efficient", "ocys-net", "swin-transformer", "yolo", "vision-transformer", 'maxvit',
    "medsam1"
]

def model_Loader(model_name: str, outlayer_num: int, type: str):
    """ 
    Model Loader : 정해진 모델 가지고 오는 로더
    만약 model이 없으면 에러 발생시키고, 있으면 모델명과 같은 파이썬 파일에서 모델 가지고 오기 
    type : 모델의 타입 (예: 'binary', 'multi')
    """
    if model_name in MODEL_LIST:
        try:
            # 모델 이름에서 하이픈을 언더스코어로 변환하여 모듈 이름으로 사용
            module_name = model_name.replace('-', '_')
            # 전체 모듈 경로 지정 (예: 'model.vgg')
            full_module_name = f"model.{module_name}"
            # 동적으로 모듈 임포트
            module = importlib.import_module(full_module_name)
            
            # 모듈에서 함수 가져오기
            if outlayer_num > 1:
                model_fn = getattr(module, "multi_model")
            elif outlayer_num == 1: 
                model_fn = getattr(module, "binary_model")
            else: 
                raise ValueError(f"outlayer_num must be greater than 0.")
            
            print(f"Model '{model_name}' loaded.")
            return model_fn(type)  # 'type'을 전달하여 모델 인스턴스 반환
                
        except ImportError:
            raise ImportError(f"모듈 '{full_module_name}'을(를) 임포트할 수 없습니다.")
        except AttributeError:
            raise ImportError(f"모델 함수 'multi_model' 또는 'binary_model'을(를) '{full_module_name}' 모듈에서 찾을 수 없습니다.")
        except Exception as e:
            raise RuntimeError(f"모델 로딩 중 오류 발생: {e}")
    else:
        raise ValueError(f"Model '{model_name}' not found in MODEL_LIST.")
