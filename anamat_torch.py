import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
import torch.nn as nn
import random
path = 'Data.mat'
data = sio.loadmat(path)
delays = data['Delays'][5:135, 0].reshape((130, 1))
ROI = data['ROIintNorm'][0, 0]['mean']
Qx = data['QxNorm'][0, 0]['mean']
Qy = data['QyNorm'][0, 0]['mean']
Qr = data['QrNorm'][0, 0]['mean']
I2 = ROI[5:135, 2].reshape((130, 1))

k, s0, xi_g, l, beta, tau, A, tdamp, Period =[-0.014552258365648323, 0.10633795406511092, 20.18595495253185, 12.147827596447192, 0.021517254204924254, 1466.6548495398872, 0.2702387912251527, 1261.7889322740823, 204.03711616311742]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.k = nn.Parameter(torch.tensor([[k]], requires_grad=True).double())
        self.s0 = nn.Parameter(torch.tensor(
            [[s0]], requires_grad=True).double())
        self.xi_g = nn.Parameter(torch.tensor(
            [[xi_g]], requires_grad=True).double())
        self.l = nn.Parameter(torch.tensor([[l]], requires_grad=True).double())
        self.beta = nn.Parameter(torch.tensor(
            [[beta]], requires_grad=True).double())
        self.tau = nn.Parameter(torch.tensor(
            [[tau]], requires_grad=True).double())
        self.A = nn.Parameter(torch.tensor([[A]], requires_grad=True).double())
        self.tdamp = nn.Parameter(torch.tensor(
            [[tdamp]], requires_grad=True).double())
        self.Period = nn.Parameter(torch.tensor(
            [[Period]], requires_grad=True).double())

    def forward(self, x):
        d = torch.exp(x.neg().div(self.tdamp)).mul(torch.cos(x.mul(2*math.pi).div(self.Period)).mul(self.A))
        s = d.mul(self.k).add(self.s0)
        frac2 = (s.mul(self.xi_g)).pow(2).add(1)
        frac1=torch.sin(torch.sqrt(frac2).div(
            self.xi_g).mul(self.l).mul(math.pi)).pow(2)
        frac3=torch.exp(x.neg().div(self.tau)).mul(
            self.beta).add(1).sub(self.beta)
        I=frac1.mul(frac3).div(frac2)
        I0=I[0]
        output=I.sub(I0).div(I0)
        return output

x=torch.from_numpy(delays).double().requires_grad_()
y=torch.from_numpy(I2).double()

model=Net()
criterion=torch.nn.MSELoss(reduction='sum')
optimizer=torch.optim.Adam(model.parameters())#, lr=1e-4)

bestloss=1
for i in range(3000000):
    y_pred=model(x)
    loss=criterion(y_pred, y)
    if i % 10000 == 0:
        print(i, loss.item())
        if loss.item() < bestloss:
            print([model.k.item(), model.s0.item(), model.xi_g.item(), model.l.item(), model.beta.item(
            ), model.tau.item(), model.A.item(), model.tdamp.item(), model.Period.item()])
            bestloss=loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
