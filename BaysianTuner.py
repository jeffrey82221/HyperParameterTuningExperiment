from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from XORTrainerFunc import xor_trainer_function


def target_function(*input):
  val_acc, epochs = xor_trainer_function(*input)
  return val_acc - epochs


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
    n_iter=8,      # also determine by the boundary of each parameter
)
# access history and result

for i, res in enumerate(optimizer.res):
  print("Iteration {}: \n\t{}".format(i, res))

print("Final Max:", optimizer.max)
