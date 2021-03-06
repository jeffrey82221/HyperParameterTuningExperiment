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
from ray.tune import register_trainable

register_trainable("target_function", target_function)
ray.init(num_cpus=48, num_gpus=3)
'''
sched = AsyncHyperBandScheduler(
    time_attr="training_iteration",
    metric="mean_acc",
    mode="max",
    max_t=400,
    grace_period=20)
'''

analysis = tune.run(
    target_function,
    name="xor",
    stop={
        "mean_accuracy": 0.99,
        "training_iteration": 3,
    },
    num_samples=30,
    max_failures=5,
    config={
        "batch_size_continuous": tune.grid_search([16384]),  # tune.sample_from(lambda spec: np.random.uniform(4, 4000)),
        "lr_exp": tune.sample_from(lambda spec: np.random.uniform(1.1, 2)),
        "momentum": tune.sample_from(
            lambda spec: np.random.uniform(0.1, 0.9)),
        "layer_size_continuous": tune.grid_search([32]),
        "layer_count_continuous": tune.grid_search([1]),
    },
    resources_per_trial={
        "cpu": 1,
        "gpu": 1,
        "extra_gpu":0
    },
    queue_trials = True
)
print("Best trail is", analysis.get_best_trial(metric="mean_accuracy", mode='max', scope='all'))
print("Best config is", analysis.get_best_config(metric="mean_accuracy", mode='max', scope='all'))
#print(analysis.dataframe(metric='mean_accuracy', mode='max'))
#print(analysis.dataframe(metric='training_iteration', mode='max'))
