from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from XORTrainerFunc import xor_trainer_function


def target_function(batch_size_continuous, lr_exp, momentum,
                    layer_size_continuous, layer_count_continuous):
  val_acc, training_time = xor_trainer_function(batch_size_continuous, lr_exp,
                                                momentum, layer_size_continuous,
                                                layer_count_continuous)
  print("val_acc:", val_acc, "training time:", training_time)
  return val_acc - training_time


pbounds = {
    'batch_size_continuous': (4, 4000),
    'lr_exp': (-2, 2),
    'momentum': (0.8, 0.999),
    'layer_size_continuous': (1, 5),
    'layer_count_continuous': (1, 3)
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
    init_points=300,  # determine according to the boundary of each parameter
    n_iter=8,  # also determine by the boundary of each parameter
)
# access history and result

for i, res in enumerate(optimizer.res):
  print("Iteration {}: \n\t{}".format(i, res))

print("Final Max:", optimizer.max)
