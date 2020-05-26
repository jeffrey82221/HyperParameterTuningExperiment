from XORTrainable import InitialAccuracyTrainable
import ray
from ray.tune import grid_search, sample_from
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune import register_trainable
from ray.tune import run
import numpy as np
register_trainable('InitialAccuracyTrainable', InitialAccuracyTrainable)

ray.init(num_cpus=40, num_gpus=0)
pbt = PopulationBasedTraining(
    time_attr="training_iteration", metric='mean_loss',
    mode='min',
    perturbation_interval=4,
    hyperparam_mutations={
        'lr_exp': lambda : np.random.uniform(-3., 3.),
        'momentum': lambda : np.random.uniform(0., 1.0)
    })
analysis = run(
    InitialAccuracyTrainable,
    name="pbt_test",
    scheduler=pbt,
    reuse_actors=True,
    checkpoint_freq=2,
    # keep_checkpoints_num=100,
    #checkpoint_score_attr = "min-mean_loss"
    verbose=True,
    stop={
        "training_iteration": 8,
    },
    resources_per_trial={
        "cpu": 1,
        "gpu": 0
    },
    sync_on_checkpoint=True,
#sync_to_driver = False, 
    num_samples=30,
    config={
        'epochs': 8,
        'batch_size_continuous': 4,
        'lr_exp': sample_from(lambda spec:np.random.uniform(-2.,2.)),
        'momentum': sample_from(lambda spec: np.random.uniform(0.1, 0.8)),
        'layer_size_continuous': grid_search([2, 4, 8, 16, 32, 64, 128, 256]),
        'layer_count_continuous': grid_search([1, 2, 3])
    },
    local_dir=".",
    queue_trials = True
    # restore="./ray_results_xor"
)
print("Best Loss Config", analysis.get_best_config(metric="mean_loss", mode='min', scope='all'))
print("Best Accuracy Config", analysis.get_best_config(metric="mean_accuracy", mode='max', scope='all'))
