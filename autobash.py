import scipy.io as sio
import numpy as np
from numba import jit
import torch
import math
import torch.nn as nn
import random
path = 'Data.mat'
data = sio.loadmat(path)
delays = data['Delays'][5:, 0]
Timezero = delays[5]
ROI = data['ROIintNorm'][0, 0]['mean']
Qx = data['QxNorm'][0, 0]['mean']
Qy = data['QyNorm'][0, 0]['mean']
Qr = data['QrNorm'][0, 0]['mean']
I2 = ROI[5:, 2]
Qr2 = Qr[:, 2]


@jit
def Dfunction(A,tdamp,Period):
    delta_d = A*np.exp(-delays/tdamp)*np.cos(2*np.pi*delays/Period)
    return delta_d


@jit
def sfunction(k, s0,A,tdamp,Period):
    s = k*Dfunction(A,tdamp,Period)+s0
    return s

@jit
def Ifunction(xi_g, l, beta, tau, k, s0, A,tdamp,Period):
    frac2 = 1+(xi_g*sfunction(k,s0,A,tdamp,Period))**2
    frac1 = np.sin(np.pi*l*np.sqrt(frac2)/xi_g)**2
    frac3 = (1-beta+beta*np.exp(-delays/tau))
    I = frac1*frac3/frac2
    return I

@jit
def output(xi_g, l, beta, tau, k, s0,A,tdamp,Period):
    I = Ifunction(xi_g, l, beta, tau, k, s0,A,tdamp,Period)
    I0 = I[0]
    ratio = (I - I0)/I0
    return ratio
@jit
def lossfunction(xi_g,l,beta,tau,k,s0,A,tdamp,Period):
    distance = (output(xi_g, l, beta, tau, k, s0,A,tdamp,Period)-I2)*100
    loss = np.sum(np.power(distance,2))
    return loss



k = np.random.uniform(-5,5)
s0 = np.random.uniform(-0.1,0.1)
xi_g = np.random.uniform(10,200)
l = np.random.uniform(20,50)
beta = np.random.uniform(0,1)
tau = np.random.uniform(1,100)
A = np.random.uniform(0,1)
tdamp = np.random.uniform(0,600)
Period = np.random.uniform(50,100)
T = 10000
Tmin = 1e-5
numin = 100
while T>=Tmin:
    i = 0
    while i <= numin:
        y = lossfunction(xi_g,l,beta,tau,k,s0,A,tdamp,Period)
        kNew = k+np.random.uniform(-0.0005,0.0005)*T/10
        s0New = s0+np.random.uniform(-1e-5,1e-5)*T/10
        xi_gNew = xi_g+np.random.uniform(-0.02,0.02)*T/10
        lNew = l+np.random.uniform(-0.003,0.003)*T/10
        betaNew = beta+np.random.uniform(-0.005,0.005)*T/10
        tauNew = tau+np.random.uniform(-0.01,0.01)*T/10
        ANEw = A+np.random.uniform(-1e-6,1e-6)*T/10
        tdampNew  = tdamp+np.random.uniform(-0.06,0.06)*T/10
        PeriodNew = Period+np.random.uniform(-0.003,0.003)*T/10
        if (10<lNew  and -0.5<s0New and 0.5>s0New and 10<xi_gNew and 200>xi_gNew and 0<betaNew and 1>betaNew and 1<tauNew  and tdampNew>0 and PeriodNew>50 and PeriodNew<120 and ANEw>0):  #and 0<kNew and 1>kNew  
            yNew = lossfunction(xi_gNew,lNew,betaNew,tauNew,kNew,s0New,ANEw,tdampNew,PeriodNew)
            if yNew<y:
                k = kNew
                s0 =s0New
                xi_g = xi_gNew
                l = lNew
                beta = betaNew
                tau = tauNew
                A = ANEw
                tdamp = tdampNew
                Period = PeriodNew
                i = i+1
            else:
                p = np.exp((y-yNew)/T)
                r = np.random.uniform(0,1)
                if r<p:
                    k = kNew
                    s0 =s0New
                    xi_g = xi_gNew
                    l = lNew
                    beta = betaNew
                    tau = tauNew
                    A = ANEw
                    tdamp = tdampNew
                    Period = PeriodNew
                    i = i+1
            
    T = 0.999*T
    print(T)
    print(y)
print([k,s0,xi_g,l,beta,tau,A,tdamp,Period])

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
optimizer=torch.optim.Adam(model.parameters())#, lr=5e-4)

bestloss=1
for i in range(3000000):
    y_pred=model(x)
    loss=criterion(y_pred, y)
    if i % 10000 == 0:
        print(i, loss.item())
        if loss.item() < bestloss:
            result = [model.k.item(), model.s0.item(), model.xi_g.item(), model.l.item(), model.beta.item(
            ), model.tau.item(), model.A.item(), model.tdamp.item(), model.Period.item()]
            print(result)
            file = open('explog.txt','a')
            file.write(str(loss.item())+'\n')
            file.write(str(result)+'\n')
            file.close()
            bestloss=loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()