"""
---
title: Noam optimizer from Attention is All You Need paper
summary: >
  This is a tutorial/implementation of Noam optimizer.
  Noam optimizer has a warm-up period and then an exponentially decaying learning rate.
---

# Noam Optimizer

This is the [PyTorch](https://pytorch.org) implementation of optimizer introduced in the paper
[Attention Is All You Need](https://arxiv.org/abs/1706.03762).
"""
from typing import Dict
import torch
from src.training.scheduler import NoamScheduler


def _test_noam_lr():
    """
    ### Plot learning rate for different warmups and model sizes

    ![Plot of learning rate](noam_lr.png)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from torch import nn

    model = nn.Linear(10, 10)
    # Create separate optimizers/schedulers so each schedule is independent
    sched_configs = [(1024, 16000, 0.1), (1024, 8000, 0.5), (1024, 8000, 0.1), (2048, 2000, 0.1)]
    optimizers = []
    schedulers = []
    for d_model, warmup, lr in sched_configs:
      optimizer = torch.optim.Adam(model.parameters(), lr=lr)
      sched = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup, factor=lr)
      optimizers.append(optimizer)
      schedulers.append(sched)

    steps = 20000
    lrs = [[] for _ in schedulers]
    for _ in range(1, steps + 1):
      for idx, (opt, sched) in enumerate(zip(optimizers, schedulers)):
        # advance scheduler one step and record current optimizer LR
        sched.step()
        lrs[idx].append(opt.param_groups[0]['lr'])

    x = np.arange(1, steps + 1)
    for lr in lrs:
      plt.plot(x, lr)
    plt.legend(["lr-0.1-1024:2000", "lr-0.5-1024:8000", "lr-0.1-1024:8000", "lr-0.1-2048:2000"])
    plt.title("Learning Rate")
    plt.show()


if __name__ == '__main__':
    _test_noam_lr()