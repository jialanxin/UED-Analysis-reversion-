import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
path = 'Data.mat'
data = sio.loadmat(path)
delays = data['Delays'][5:80, 0]
Timezero = delays[5]
ROI = data['ROIintNorm'][0, 0]['mean']
Qx = data['QxNorm'][0, 0]['mean']
Qy = data['QyNorm'][0, 0]['mean']
Qr = data['QrNorm'][0, 0]['mean']
I2 = ROI[5:80, 2]


def lossfunction(lr,w1,w2):
    loss = np.random.uniform()
    return -loss



bo = BayesianOptimization(lossfunction, {'lr': (1e-7, 3.0), 'w1': (1e-7,3.0), 'w2': (1e-7, 3.0)})

bo.initialize(
{
    'target':[-7.370537564667519,-27.59159308431763,-8.702170722658967,-8.630021750885921,-10000,-9.743792334985518,-27.313756237878645,-8.487234460185377,-8.536150569285885],
    'lr':[1.1,3.0,0.0,0.0,0.0,3.0,3.0,1.6560,1.1576],
    'w1':[1.49445,3.0,0.0,3.0,0.0,1.758,0.0,3.0,1.5129],
    'w2':[1.49445,3.0,0.0,0.0,3.0,0.0,0.6952,1.0951,0.0]
}
)
bo.maximize(init_points=0, n_iter=1, acq='ucb', kappa=5)
