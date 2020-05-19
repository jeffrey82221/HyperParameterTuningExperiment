from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from XORTrainerFunc import initial_acc  # initial_slop
'''
pbounds = {
    'batch_size_continuous': (4, 4000),
    'lr_exp': (-0.5, 0.5),
    'momentum': (0.85, 0.99),
    'layer_size_continuous': (2, 5),
    'layer_count_continuous': (1.5, 2.5)
}
'''
pbounds = {
    'batch_size_continuous': (1500, 2000),
    'lr_exp': (0.5, 1.5),
    'momentum': (0.9, 0.99),
    'layer_size_continuous': (5, 10),
    'layer_count_continuous': (1, 1)
}
optimizer = BayesianOptimization(
    f=initial_acc,
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
    n_iter=1000,  # also determine by the boundary of each parameter
    init_points=2**5,  # determine according to the boundary of each parameter
)
# access history and result

for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))

print("Final Max:", optimizer.max)
