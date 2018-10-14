import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
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
def Dfunction(A,tdamp,Period,phi):
    delta_d = A*np.exp(-delays/tdamp)*np.cos(2*np.pi*delays/Period+phi)
    return delta_d


@jit
def sfunction(k, s0,A,tdamp,Period,phi):
    s = k*Dfunction(A,tdamp,Period,phi)+s0
    return s

@jit
def Ifunction(xi_g, l, beta, tau, k, s0, A,tdamp,Period,phi):
    frac2 = 1+(xi_g*sfunction(k,s0,A,tdamp,Period,phi))**2
    frac1 = np.sin(np.pi*l*np.sqrt(frac2)/xi_g)**2
    frac3 = (1-beta+beta*np.exp(-delays/tau))
    I = frac1*frac3/frac2
    return I

@jit
def output(xi_g, l, beta, tau, k, s0,A,tdamp,Period,phi):
    I = Ifunction(xi_g, l, beta, tau, k, s0,A,tdamp,Period,phi)
    I0 = I[0]
    ratio = (I - I0)/I0
    return ratio
@jit
def lossfunction(xi_g,l,beta,tau,k,s0,A,tdamp,Period,phi):
    distance = (output(xi_g, l, beta, tau, k, s0,A,tdamp,Period,phi)-I2)*100
    loss = np.sum(np.power(distance,2))
    return loss

k = np.random.uniform(-5,5)
s0 = np.random.uniform(-0.1,0.1)
xi_g = np.random.uniform(10,200)
l = np.random.uniform(20,50)
beta = np.random.uniform(0,1)
tau = np.random.uniform(1,100)
A = np.random.uniform(0,0.1)
tdamp = np.random.uniform(0,600)
Period = np.random.uniform(50,80)
phi = np.random.uniform(-np.pi,np.pi)







T = 10000
Tmin = 1e-10
numin = 100
while T>=Tmin:
    i = 0
    while i <= numin:
        y = lossfunction(xi_g,l,beta,tau,k,s0,A,tdamp,Period,phi)
        kNew = k+np.random.uniform(-0.0005,0.0005)*T
        s0New = s0+np.random.uniform(-1e-5,1e-5)*T
        xi_gNew = xi_g+np.random.uniform(-0.02,0.02)*T
        lNew = l+np.random.uniform(-0.003,0.003)*T
        betaNew = beta+np.random.uniform(-0.005,0.005)*T
        tauNew = tau+np.random.uniform(-0.01,0.01)*T
        ANEw = A+np.random.uniform(-1e-6,1e-6)*T
        tdampNew  = tdamp+np.random.uniform(-0.06,0.06)*T
        PeriodNew = Period+np.random.uniform(-0.003,0.003)*T
        phiNew = phi+np.random.uniform(-np.pi/10000,np.pi/10000)*T
        if (20<lNew and 50>lNew  and -0.5<s0New and 0.5>s0New and 10<xi_gNew and 200>xi_gNew and 0<betaNew and 1>betaNew and 1<tauNew and 100> tauNew and tdampNew>0 and PeriodNew>50 and PeriodNew<80 and phiNew>-np.pi and phiNew<np.pi):  #and 0<kNew and 1>kNew  
            yNew = lossfunction(xi_gNew,lNew,betaNew,tauNew,kNew,s0New,ANEw,tdampNew,PeriodNew,phiNew)
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
                phi = phiNew
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
                    phi = phiNew
            i = i+1
    T = 0.999*T
    print(T)
    print(y)
    
print([k,s0,xi_g,l,beta,tau,A,tdamp,Period,phi])