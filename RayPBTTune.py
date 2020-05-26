from XORTrainable import InitialAccuracyTrainable
import ray
from ray.tune import grid_search
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune import register_trainable
from ray.tune import run
import numpy as np
register_trainable('InitialAccuracyTrainable', InitialAccuracyTrainable)

ray.init(num_cpus=40)
pbt = PopulationBasedTraining(
    time_attr="training_iteration", metric='mean_loss',
    mode='min',
    perturbation_interval=10,
    hyperparam_mutations={
        'lr_exp': lambda _: np.random.uniform(-3., 3.),
        'momentum': lambda _: np.random.uniform(0., 1.)
    })
analysis = run(
    InitialAccuracyTrainable,
    name="pbt_test",
    scheduler=pbt,
    reuse_actors=True,
    checkpoint_freq=2,
    keep_checkpoints_num=4,
    verbose=True,
    stop={
        "training_iteration": 5,
    },
    resources_per_trial={
        "cpu": 1,
        "gpu": 0
    },
    num_samples=100,
    config={
        'epochs': 5,
        'batch_size_continuous': 4,
        'lr_exp': 0,
        'momentum': 0.5,
        'layer_size_continuous': grid_search([2, 4, 8, 16, 32, 64, 128, 256]),
        'layer_count_continuous': grid_search([1, 2, 3])
    },
    local_dir="./ray_results_xor"
)
