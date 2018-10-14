import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
import torch.nn as nn
import random
path = 'Data.mat'
data = sio.loadmat(path)
delays = data['Delays'][5:135, 0].reshape((130,1))
ROI = data['ROIintNorm'][0, 0]['mean']
Qx = data['QxNorm'][0, 0]['mean']
Qy = data['QyNorm'][0, 0]['mean']
Qr = data['QrNorm'][0, 0]['mean']
I2 =ROI[5:135, 2].reshape((130,1))
Qr2 = Qr[5:135, 2].reshape((130,1))

#k,s0,xi_g,l,beta,tau=[-2.871997261867213, 0.042938014328291474, 152.68584852294265, 32.24929553024363, 1.4558021412360844e-15, 70.02938890733687]
k,s0,xi_g,l,beta,tau,A,tdamp,Period,phi=[0.04524547728640656, 0.11700898847432833, 37.015771642528364, 20.632806300039096, 0.02166922706447096, 825.0083792679802, 0.0452451844858905, 88.43765967844969, 82.91777927425903, -0.7105963646864378]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.A = nn.Parameter(torch.tensor([[A]],requires_grad = True).double())	
    def forward(self,x):
        d = torch.exp(x.neg().div(tdamp)).mul(torch.cos(x.mul(2*math.pi).div(Period).add(phi))).mul(self.A)
        return d

x = torch.from_numpy(delays).double().requires_grad_()
y = torch.from_numpy(Qr2).double()

model = Net()
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-6)

bestloss = 1
for i in range(500000):
    y_pred = model(x)
    loss = criterion(y_pred,y)
    if i%10000 == 0:
        print(i, loss.item())
        if loss.item()<bestloss:
            print(model.A.item())
            bestloss = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


















