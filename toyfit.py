import scipy.io as sio
import numpy as np
import torch
import math
import torch.nn as nn
import random
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
path = 'Data.mat'
data = sio.loadmat(path)
delays = data['Delays'][5:80, 0].reshape((75,1))
ROI = data['ROIintNorm'][0, 0]['mean']
Qx = data['QxNorm'][0, 0]['mean']
Qy = data['QyNorm'][0, 0]['mean']
Qr = data['QrNorm'][0, 0]['mean']
I2 = ROI[5:80, 2].reshape((75, 1))
Qr2 = Qr[5:80,2].reshape((75,1))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.inputlayer = nn.Linear(1,5)
        self.hiddenlayer1 = nn.Linear(5,5)
        self.hiddenlayer2 = nn.Linear(5,5)
        self.hiddenlayer3 = nn.Linear(5,5)
        self.hiddenlayer4 = nn.Linear(5,5)
        self.hiddenlayer5 = nn.Linear(5,5)
        self.hiddenlayer6 = nn.Linear(5,5)
        self.hiddenlayer7 = nn.Linear(5,5)
        self.hiddenlayer8 = nn.Linear(5,5)
        self.hiddenlayer9 = nn.Linear(5,5)
        self.hiddenlayer10 = nn.Linear(5,5)
        self.hiddenlayer11 = nn.Linear(5,5)
        self.hiddenlayer12 = nn.Linear(5,5)
        self.hiddenlayer13 = nn.Linear(5,5)
        self.hiddenlayer14 = nn.Linear(5,5)
        self.hiddenlayer15 = nn.Linear(5,5)
        self.outputlayer = nn.Linear(5,1)
    def forward(self,x):
        x = self.inputlayer(x).clamp(min=0)
        x = self.hiddenlayer1(x).clamp(min=0)
        x = self.hiddenlayer2(x).clamp(min=0)
        x = self.hiddenlayer3(x).clamp(min=0)
        x = self.hiddenlayer4(x).clamp(min=0)
        x = self.hiddenlayer5(x).clamp(min=0)
        x = self.hiddenlayer6(x).clamp(min=0)
        x = self.hiddenlayer7(x).clamp(min=0)
        x = self.hiddenlayer8(x).clamp(min=0)
        x = self.hiddenlayer9(x).clamp(min=0)
        x = self.hiddenlayer10(x).clamp(min=0)
        x = self.hiddenlayer11(x).clamp(min=0)
        x = self.hiddenlayer12(x).clamp(min=0)
        x = self.hiddenlayer13(x).clamp(min=0)
        x = self.hiddenlayer14(x).clamp(min=0)
        x = self.hiddenlayer15(x).clamp(min=0)
        y_pred = self.outputlayer(x)
        return y_pred

net = Net()
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(net.parameters())
x = delays
x_mean = np.mean(x)
x_var = np.var(x)
x = (x-x_mean)/np.sqrt(x_var)
x = torch.from_numpy(x).float().requires_grad_()
y_val = I2
y_val_mean = np.mean(I2)
y_val_var = np.var(I2)
y_val = (y_val-y_val_mean)/np.sqrt(y_val_var)
y_val = torch.from_numpy(y_val).float()

for i in range(40000):
    data = np.hstack(delays,I2)
    mask = np.random.randint(2,size)
    y_pred = net(x)
    loss = criterion(y_pred,y_val)
    if i%1000==0:
        print(i,loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(net.state_dict(),'model.pt')
y_pred = net(x).detach().numpy().reshape(75,)*np.sqrt(y_val_var)+y_val_mean
plt.figure()
plt.plot(delays,y_pred)
plt.plot(delays,I2)
plt.show()