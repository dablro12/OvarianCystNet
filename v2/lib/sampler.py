import torch
import random
import math

class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, labels=None, batch_size=4):
        """
        균형 잡힌 배치를 생성하기 위한 샘플러.

        Args:
            dataset (Dataset): PyTorch Dataset 객체.
            labels (Tensor, optional): 레이블 텐서. 제공되지 않으면 데이터셋에서 레이블을 추출.
            batch_size (int): 배치 크기. 클래스 수로 나누어 떨어져야 함.
        """
        self.labels = labels
        self.dataset = dict()
        self.batch_size = batch_size

        # 클래스별 인덱스 저장
        for idx in range(len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = []
            self.dataset[label].append(idx)
        self.keys = list(self.dataset.keys())
        
        self.num_classes = len(self.keys)
        print(f"self.batch_size : {self.batch_size} | self.num_classes : {self.num_classes}")
        if self.batch_size % self.num_classes != 0:
            raise ValueError("Batch size must be divisible by number of classes")

        self.samples_per_class = self.batch_size // self.num_classes

        # 각 클래스의 샘플 수 중 최대값 찾기
        original_max = max(len(indices) for indices in self.dataset.values())
        # balanced_max를 samples_per_class의 최소 배수로 설정
        self.balanced_max = math.ceil(original_max / self.samples_per_class) * self.samples_per_class

        # 클래스별 인덱스 오버샘플링
        for label in self.dataset:
            current_len = len(self.dataset[label])
            if current_len < self.balanced_max:
                num_to_add = self.balanced_max - current_len
                # 무작위로 샘플 복제
                self.dataset[label].extend(random.choices(self.dataset[label], k=num_to_add))

        # 모든 클래스가 balanced_max 개의 샘플을 가지는지 확인
        for label, indices in self.dataset.items():
            assert len(indices) == self.balanced_max, f"Class {label} has {len(indices)} samples, expected {self.balanced_max}"

    def __iter__(self):
        """
        샘플러의 이터레이터.

        Returns:
            Iterator of indices.
        """
        # 각 클래스의 인덱스 무작위로 섞기
        shuffled_dataset = {label: random.sample(indices, len(indices)) for label, indices in self.dataset.items()}

        # 총 배치 수 계산
        num_batches = self.balanced_max // self.samples_per_class

        all_batches = []

        for i in range(num_batches):
            batch = []
            for label in self.keys:
                start = i * self.samples_per_class
                end = (i + 1) * self.samples_per_class
                batch.extend(shuffled_dataset[label][start:end])
            random.shuffle(batch)  # 배치 내 샘플 순서 섞기
            all_batches.append(batch)

        random.shuffle(all_batches)  # 전체 배치 순서 섞기

        # 인덱스를 순차적으로 yield
        for batch in all_batches:
            for idx in batch:
                yield idx

    def _get_label(self, dataset, idx):
        """
        주어진 인덱스의 레이블을 반환.

        Args:
            dataset (Dataset): PyTorch Dataset 객체.
            idx (int): 데이터 인덱스.

        Returns:
            레이블 값.
        """
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            # 기본 레이블 추출 방식 (사용자 정의 데이터셋에 맞게 수정 필요)
            if hasattr(dataset, 'labels'):
                return dataset.labels[idx]
            elif hasattr(dataset, 'targets'):
                return dataset.targets[idx]
            elif hasattr(dataset, 'imgs'):
                return dataset.imgs[idx][1]
            else:
                raise AttributeError("Dataset에 레이블을 추출할 수 있는 속성이 없습니다. labels 인자를 제공하세요.")

    def __len__(self):
        """
        샘플러가 생성할 총 샘플 수.

        Returns:
            int: 총 샘플 수.
        """
        return self.balanced_max * self.num_classes


#%% [CycleGAN] Sampler
class IterationBasedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    Wraps a BatchSampler, re-sampling from it until [num_iterations] iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations