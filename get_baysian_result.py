from XORTrainerFunc import xor_trainer_function
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
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
min_target = 0


def target_function(batch_size_continuous, lr_exp, momentum,
                    layer_size_continuous, layer_count_continuous):
    global min_target
    val_acc, training_time = xor_trainer_function(batch_size_continuous, lr_exp,
                                                  momentum, layer_size_continuous,
                                                  layer_count_continuous)

    target = val_acc * 30 - training_time
    if val_acc == 0.5:
        target = min_target
    if target < min_target:
        min_target = target
    print("val_acc:", val_acc, "training time:", training_time, 'target:', target)
    return target


def get():
    pbounds = {
        'batch_size_continuous': (4, 4000),
        'lr_exp': (-2, 2),
        'momentum': (0.8, 0.999),
        'layer_size_continuous': (1, 5),
        'layer_count_continuous': (1, 3)
    }
    optimizer = BayesianOptimization(
        f=target_function,
        pbounds=pbounds
    )
    load_logs(optimizer, logs=["./baysian_logs.json"])

    res_list = []
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
        res_list.append(res)

    # print("Final Max:", optimizer.max)
    return res_list
