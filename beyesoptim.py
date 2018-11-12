import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization


def lossfunction(lr,w1,w2):
    loss = np.random.uniform()*20
    return -loss



bo = BayesianOptimization(lossfunction, {'lr': (0.3, 3.0), 'w1': (0.3,3.0), 'w2': (0.3, 3.0)})

bo.initialize(
{
    'target':[-16.626141084222657,-16.656124820118073,-16.62569318934877,-16.62492867148469],
    'lr':[1.6879,2.1292,0.3,0.3],
    'w1':[3.0000,2.6128,0.3,0.3],
    'w2':[0.9983,2.0528,0.3,3.0]
}
)
bo.maximize(init_points=0, n_iter=2, acq='ucb', kappa=2)
