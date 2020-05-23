from XORTrainerFunc import initial_acc  # initial_slop
from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs

'''
from get_baysian_result import get
data = get()
for v in list(map(lambda x:x['target'],data)):
    print(v)
for v in list(map(lambda x:x['params']['batch_size_continuous'],data)):
    print(v)
for v in list(map(lambda x:x['params']['lr_exp'],data)):
    print(v)
for v in list(map(lambda x:x['params']['momentum'],data)):
    print(v)
for v in list(map(lambda x:x['params']['layer_size_continuous'],data)):
    print(v)
for v in list(map(lambda x:x['params']['layer_count_continuous'],data)):
    print(v)
 '''


def get():
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
    optimizer = BayesianOptimization(
        f=initial_acc,  # initial_slop,
        pbounds=pbounds
    )
    load_logs(optimizer, logs=["./baysian_logs.json"])

    res_list = []
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
        res_list.append(res)

    # print("Final Max:", optimizer.max)
    return res_list
