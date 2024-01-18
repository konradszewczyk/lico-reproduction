# from https://gist.github.com/akshaychawla/86d938bc6346cf535dce766c83f743ce

import torch
import math
import functools


def _cosine_decay(iteration, total_iterations):
    multiplier = math.cos(7 * math.pi * iteration / (16 * total_iterations))
    return multiplier


def CosineLRScheduler(optimizer, T_max):
    _decay_func = functools.partial(
        _cosine_decay,
        total_iterations=T_max
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    return scheduler


if __name__ == "__main__":
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Dummy parameters
    parameters = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    total_iters = 1000

    # Test CosineAnnealingLRWarmup
    optimizer = torch.optim.Adam([parameters], lr=1)
    scheduler = CosineLRScheduler(optimizer, T_max=total_iters)
    actual_lr = []
    for _iter in range(total_iters):
        scheduler.step()
        actual_lr.append(optimizer.param_groups[0]["lr"])
    plt.plot(list(range(total_iters)), actual_lr, label="CosineLRScheduler")

    plt.xlabel("iterations")
    plt.ylabel("learning rate")
    plt.legend()
    plt.savefig("scheduler.png")
