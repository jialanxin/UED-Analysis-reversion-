import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
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


class ParticleGroup():
    def __init__(self,group_size,weight,c1,c2):
        self.group_size = group_size
        self.x_cur_group = self.ramdom_generator().double()
        self.v_cur_group = self.x_cur_group.div(5)
        print('v_cur finished')
        self.x_down_group = torch.Tensor([-10.0,-0.1,10.0,20.0,1e-7,1e-7,1e-7,1e-7,50.0]).repeat((group_size,1)).double()
        self.x_up_group = torch.Tensor([10.0,0.1,200.0,60.0,1.0,100.0,1.0,600.0,120.0]).repeat((group_size,1)).double()
        print('x bound finished')
        self.v_up_group = self.x_up_group.div(5)
        self.v_down_group = self.v_up_group.neg()
        print('v bound finished')
        self.input_data = torch.from_numpy(delays).repeat((group_size,1)).double()
        self.validation_data = torch.from_numpy(I2).repeat((group_size,1)).double()
        print('x-y finished')
        self.loss_history_best_group = self.Lossfunction()
        print('loss finished')
        self.x_history_best_group = self.x_cur_group
        self.loss_history_best_total,self.index = torch.min(self.loss_history_best_group,0)
        self.x_history_best_total = self.x_history_best_group[self.index,:]   
        print('all finished')
        self.weight = weight
        self.c1 = c1
        self.c2 = c2
    def ramdom_generator(self):
        x_cat = torch.Tensor([[np.random.uniform(-5, 5), np.random.uniform(-0.1, 0.1), np.random.uniform(10, 200), np.random.uniform(20, 50),
                               np.random.uniform(0, 1), np.random.uniform(0, 100), np.random.uniform(0, 1), np.random.uniform(0, 600), np.random.uniform(50, 100)]])
        for i in range(self.group_size-1):
            x_cur = torch.Tensor([[np.random.uniform(-5, 5), np.random.uniform(-0.1, 0.1), np.random.uniform(10, 200), np.random.uniform(20, 50),
                               np.random.uniform(0, 1), np.random.uniform(0, 100), np.random.uniform(0, 1), np.random.uniform(0, 600), np.random.uniform(50, 100)]])
            x_cat = torch.cat((x_cat,x_cur),0)   
        print('ram gen finished')
        return x_cat
    def Lossfunction(self):
        A = self.x_cur_group[:,6].reshape((self.group_size,1))
        tdamp = self.x_cur_group[:,7].reshape((self.group_size,1))
        Period = self.x_cur_group[:,8].reshape((self.group_size,1))
        d = torch.exp(self.input_data.neg().div(tdamp)).mul(torch.cos(self.input_data.mul(2*math.pi).div(Period))).mul(A)
        k = self.x_cur_group[:,0].reshape((self.group_size,1))
        s0 = self.x_cur_group[:,1].reshape((self.group_size,1))
        s = d.mul(k).add(s0)
        xi = self.x_cur_group[:,2].reshape((self.group_size,1))
        frac1 = (s.mul(xi)).pow(2).add(1)
        l = self.x_cur_group[:,3].reshape((self.group_size,1))
        frac2 = torch.sin(torch.sqrt(frac1).div(xi).mul(l).mul(math.pi)).pow(2)
        beta = self.x_cur_group[:,4].reshape((self.group_size,1))
        tau = self.x_cur_group[:,5].reshape((self.group_size,1))
        frac3 = torch.exp(self.input_data.neg().div(tau)).mul(beta).add(1).sub(beta)
        I=frac2.mul(frac3).div(frac1)
        I0 = I[:,0].reshape((self.group_size,1))
        NI = I.sub(I0).div(I0)
        Loss = torch.sum(self.validation_data.sub(NI).pow(2),1,keepdim = True)
        return Loss
    
    def evolution(self):
        r1,r2 = torch.rand(2).double()
        self.v_cur_group = self.v_cur_group.mul(self.weight).add(self.x_history_best_group.sub(self.x_cur_group).mul(self.c1*r1)).add(self.x_history_best_total.sub(self.x_cur_group).mul(self.c2*r2))
        self.x_cur_group = self.x_cur_group.add(self.v_cur_group)
        if torch.ge(torch.rand(1),0.8):
            x_cur_variation_group = torch.rand_like(self.x_cur_group).mul(2)
            self.x_cur_group = x_cur_variation_group.mul(self.x_cur_group)
        self.x_cur_group = torch.min(self.x_cur_group,self.x_up_group)
        self.x_cur_group = torch.max(self.x_cur_group,self.x_down_group)
        self.v_cur_group = torch.min(self.v_cur_group,self.v_up_group)
        self.v_cur_group = torch.max(self.v_cur_group,self.v_down_group)
    
    def evaluate(self):
        loss_list_cur  = self.Lossfunction()
        mask1 = torch.le(loss_list_cur,self.loss_history_best_group)
        temp = torch.masked_select(self.x_cur_group,mask1)
        self.x_history_best_group = self.x_history_best_group.masked_scatter(mask1,temp)
        self.loss_history_best_group = torch.min(self.loss_history_best_group,loss_list_cur)
        loss_total_cur, index = torch.min(self.loss_history_best_group,0)
        if torch.le(loss_total_cur,self.loss_history_best_total):
            self.x_history_best_total = self.x_history_best_group[index,:]
            print(particlegroup.loss_history_best_total.item())
            print(particlegroup.x_history_best_total.tolist())
            with open('x_history_best_total.txt', 'a') as file:
                file.write(str(particlegroup.loss_history_best_total.item())+'\n')
                file.write(str(particlegroup.x_history_best_total.tolist())+'\n')

                



particlegroup = ParticleGroup(10000000,1.0,1.49445,1.49445)
for i in range(200):
    particlegroup.evolution()
    particlegroup.evaluate()








