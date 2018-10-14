import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
path = 'Data.mat'
data = sio.loadmat(path)
delays = data['Delays'][5:135, 0]
Timezero = delays[5]
ROI = data['ROIintNorm'][0, 0]['mean']
Qx = data['QxNorm'][0, 0]['mean']
Qy = data['QyNorm'][0, 0]['mean']
Qr = data['QrNorm'][0, 0]['mean']
I2 = ROI[5:135, 2]
Qr2 = Qr[5:135, 2]



def sfunction(k, s0):
    s = k*-Qr2+s0
    return s


def Ifunction(xi_g, l, beta, tau, k, s0):
    frac2 = 1+(xi_g*sfunction(k,s0))**2
    frac1 = np.sin(np.pi*l*np.sqrt(frac2)/xi_g)**2
    frac3 = (1-beta+beta*np.exp(-delays/tau))
    I = frac1*frac3/frac2
    return I


def output(xi_g, l, beta, tau, k, s0):
    I = Ifunction(xi_g, l, beta, tau, k, s0)
    I0 = I[0]
    ratio = (I - I0)/I0
    return ratio
def lossfunction(xi_g,l,beta,tau,k,s0):
    distance = (output(xi_g, l, beta, tau, k, s0)-I2)*100
    loss = np.sum(np.power(distance,2))
    return loss



#v5.0



k,s0,xi_g,l,beta,tau=[2.0862083887801988, 0.06921435754591368, 135.49237133055718, 21.002926181627675, 2.2916698521436937e-15, 60.84395309478068]
k,s0,xi_g,l,beta,tau=[1.7832475492663484, 0.0702068756740138, 135.0092513510239, 20.36917366556257, -0.016377303089071288, 13.47043174984272]
plt.figure(2)
plt.plot(delays[:],I2[:])
plt.plot(delays[:],output(xi_g,l,beta,tau,k,s0))


# k,s0,xi_g,l,beta,tau=[1.108836490712114, -0.024213003220826056, 59.34209843129593, 47.33084013988434, 6.290821173446003e-16, 40.93606251797164]
# k,s0,xi_g,l,beta,tau=[0.9222761010892914, -0.024914724446257284, 59.07448514086477, 47.23156369916682, -0.016380671252895718, 13.477873859158109]
# plt.figure(3)
# plt.plot(delays[:],I2[:])
# plt.plot(delays[:],output(xi_g,l,beta,tau,k,s0))


# k,s0,xi_g,l,beta,tau=[-0.14625978663573005, 0.15303889017228473, 38.661357556171126, 47.57459296128992, 9.111665718999016e-15, 83.91022207939467]
# k,s0,xi_g,l,beta,tau=[-0.767838365807469, 0.15512864451622974, 38.634604435700815, 47.55314469758121, -0.01637407282427846, 13.54778772629545]
# k,s0,xi_g,l,beta,tau=[-16.23157634944479, 0.2933689939003153, 0.12320075230955027, 0.12409527326476569, 0.05581523125133249, 0.02292362202722683]
# plt.figure(4)
# plt.plot(delays[:],I2[:])
# plt.plot(delays[:],output(xi_g,l,beta,tau,k,s0))
# Ap = 0.000526022188918724
# k,s0,xi_g,l,beta,tau,A,tdamp,Period,phi=[0.04524547728640656, 0.11700898847432833, 37.015771642528364, 20.632806300039096, 0.02166922706447096, 825.0083792679802, 0.0452451844858905, 88.43765967844969, 82.91777927425903, -0.7105963646864378]#4.6
# kp=-k*A/Ap
# k = kp
# plt.figure(4)
# plt.plot(delays[:],I2[:])
# plt.plot(delays[:],output(xi_g,l,beta,tau,k,s0))



plt.show()
