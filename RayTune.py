'''from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from XORTrainerFunc import initial_slop  # memory_efficiency  # perfect_acc_time  # model_contruction_time  # perfect_acc_time  # initial_acc  # initial_slop
'''

from ray.tune.integration.keras import TuneReporterCallback

'''
pbounds = {
    'batch_size_continuous': (4, 4000),
    'lr_exp': (-0.5, 0.5),
    'momentum': (0.85, 0.99),
    'layer_size_continuous': (2, 5),
    'layer_count_continuous': (1.5, 2.5)
}
'''


def target_function(config):
    from XORTrainerFunc import initial_slop
    batch_size_continuous = config['batch_size_continuous']
    lr_exp = config['lr_exp']
    momentum = config['momentum']
    layer_size_continuous = config['layer_size_continuous']
    layer_count_continuous = config['layer_count_continuous']
    tune_report_call_back = TuneReporterCallback()
    target = initial_slop(batch_size_continuous, lr_exp, momentum,
                          layer_size_continuous, layer_count_continuous,
                          tune_report_call_back=tune_report_call_back)
    print(
        "target:",
        target,
        "lr_exp:",
        lr_exp,
        "momentum",
        momentum
    )
    return target


import numpy as np

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune import register_trainable

register_trainable("target_function", target_function)
ray.init(num_cpus=4, num_gpus=1)
sched = AsyncHyperBandScheduler(
    time_attr="training_iteration",
    metric="mean_acc",
    mode="max",
    max_t=400,
    grace_period=20)
pbounds = {
    'batch_size_continuous': (20, 20),
    # Possibly as small as possible to reduce model construction time.
    # Effect of large batch size is the same as large lr because
    # the training batch is repeative (no variance between batches).
    'lr_exp': (1.02, 1.02),
    # As large as possible to allows larger initial gradient
    'momentum': (0.8, 0.8),
    'layer_size_continuous': (20, 20),
    # As large as possible to increase model complexity, since no overfitting is presented.)
    'layer_count_continuous': (1, 1)
    # As small as possible because large layer count leads to slower optimization.
}


tune.run(
    target_function,
    name="xor",
    scheduler=sched,
    stop={
        "mean_accuracy": 0.99,
        "training_iteration": 5,
    },
    num_samples=10,
    config={
        "batch_size_continuous": tune.sample_from(lambda spec: np.random.uniform(4, 4000)),
        "lr_exp": tune.sample_from(lambda spec: np.random.uniform(-3, 3)),
        "momentum": tune.sample_from(
            lambda spec: np.random.uniform(0.1, 0.9)),
        "layer_size_continuous": tune.grid_search([4, 16, 32, 64, 128, 256, 512]),
        "layer_count_continuous": tune.grid_search([1, 2, 3]),
    })
