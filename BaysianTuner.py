from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from XORTrainerFunc import memory_efficiency  # perfect_acc_time  # model_contruction_time  # perfect_acc_time  # initial_acc  # initial_slop
'''
pbounds = {
    'batch_size_continuous': (4, 4000),
    'lr_exp': (-0.5, 0.5),
    'momentum': (0.85, 0.99),
    'layer_size_continuous': (2, 5),
    'layer_count_continuous': (1.5, 2.5)
}
'''


def target_function(batch_size_continuous, lr_exp, momentum,
                    layer_size_continuous, layer_count_continuous):
    target = memory_efficiency(batch_size_continuous, lr_exp, momentum,
                               layer_size_continuous, layer_count_continuous)
    print(
        "target:",
        target,
        "batch_size_continuous:",
        batch_size_continuous,
        "layer_size_continuous",
        layer_size_continuous
    )
    return target


pbounds = {
    'batch_size_continuous': (100, 680),
    # Possibly as small as possible to reduce model construction time.
    # Effect of large batch size is the same as large lr because
    # the training batch is repeative (no variance between batches).
    'lr_exp': (1.2, 1.2),
    # As large as possible to allows larger initial gradient
    'momentum': (0.85, 0.85),
    'layer_size_continuous': (3, 100),
    # As large as possible to increase model complexity, since no overfitting is presented.)
    'layer_count_continuous': (1, 1)
    # As small as possible because large layer count leads to slower optimization.
}
optimizer = BayesianOptimization(
    f=target_function,
    pbounds=pbounds,
    random_state=1,
)
try:
    load_logs(optimizer, logs=["./baysian_logs.json"])
except:
    print('no baysian_logs')

# subsribing the optimizing history
logger = JSONLogger(path="./baysian_logs.json")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

# run the optimization
optimizer.maximize(
    n_iter=30,  # also determine by the boundary of each parameter
    init_points=2**2,  # determine according to the boundary of each parameter
)
# access history and result

for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))

print("Final Max:", optimizer.max)
