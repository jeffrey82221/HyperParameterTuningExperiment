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
        'batch_size_continuous': (1500, 2000),
        'lr_exp': (0.5, 1.5),
        'momentum': (0.9, 0.99),
        'layer_size_continuous': (5, 10),
        'layer_count_continuous': (1, 1)
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
