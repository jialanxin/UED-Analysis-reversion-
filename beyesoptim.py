import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization


def lossfunction(lr,w1,w2):
    loss = np.random.uniform()
    return -loss



bo = BayesianOptimization(lossfunction, {'lr': (0.3, 3.0), 'w1': (0.3,3.0), 'w2': (0.3, 3.0)})

bo.initialize(
{
    'target':[-7.551364847419092,-10,-6.74877672123009,-6.288376725887072],
    'lr':[1.0,3.0,0.3,0.3],
    'w1':[0.5,3.0,0.3,0.3],
    'w2':[1.5,3.0,0.3,3.0]
}
)
bo.maximize(init_points=0, n_iter=2, acq='ucb', kappa=2)
