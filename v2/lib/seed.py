import numpy as np 
import torch
import random
import os 

class seed_fix:
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def __exit__(self, type, value, traceback):
        pass
    

# Example
def seed_prefix(seed):
    with seed_fix(seed):
        os.environ['SEED'] = str(seed)
        print(f'Seed Fix: {seed}')
        pass