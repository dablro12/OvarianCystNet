def lr_lambda(current_step, warmup_steps = 5):
    """ 처음엔 점진적으로 학습률을 증가시키고, 이후에는 지수적으로 감소시키는 스케줄러 함수 """
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 0.95 ** (current_step - warmup_steps)