import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.multiprocessing as mp
import gc
path = 'Data.mat'
data = sio.loadmat(path)
delays = data['Delays'][5:80, 0]
Timezero = delays[5]
ROI = data['ROIintNorm'][0, 0]['mean']
Qx = data['QxNorm'][0, 0]['mean']
Qy = data['QyNorm'][0, 0]['mean']
Qr = data['QrNorm'][0, 0]['mean']
I2 = ROI[5:80, 2]
Qr2 = Qr[5:80,2]



class ParticleGroup():
    def __init__(self,group_size,weight,c1,c2,name):
        self.name = name
        self.group_size = group_size
        self.x_down_group = torch.Tensor([9e3,1e-7,10.0,1e-7,1e-7]).repeat((group_size,1)).double()
        self.x_up_group = torch.Tensor([2e4,1e6,70.0,1.0,100]).repeat((group_size,1)).double()
        self.x_cur_group = self.ramdom_generator().double()
        self.v_cur_group = self.x_cur_group.div(5)
        self.v_up_group = self.x_up_group.div(5)
        self.v_down_group = self.v_up_group.neg()
        print('v bound finished')
        self.input_Qr2 = torch.from_numpy(Qr2).repeat((group_size,1)).double()
        self.input_delays = torch.from_numpy(delays).repeat((group_size,1)).double()
        self.validation_data = torch.from_numpy(I2).repeat((group_size,1)).double()
        print('x-y finished')
        self.loss_history_best_group = self.Lossfunction()
        print('loss finished')
        self.x_history_best_group = self.x_cur_group
        self.loss_history_best_total,self.index = torch.min(self.loss_history_best_group,0)
        print(self.loss_history_best_total)
        self.x_history_best_total = self.x_history_best_group[self.index,:]   
        print(self.x_history_best_total)
        self.weight = weight
        self.c1 = c1
        self.c2 = c2
    def ramdom_generator(self):   
        x_init = torch.rand((self.group_size,5))
        x_cat = x_init.mul(self.x_up_group.float().sub(self.x_down_group.float()).add(self.x_down_group.float()))
        print('ram gen finished')
        return x_cat
    def Lossfunction(self):
        # A = self.x_cur_group[:,6].reshape((self.group_size,1))
        # tdamp = self.x_cur_group[:,7].reshape((self.group_size,1))
        # Period = self.x_cur_group[:,8].reshape((self.group_size,1))
        # d = torch.sin(self.input_delays.mul(2*math.pi).div(Period)).neg().mul(A).mul(torch.exp(self.input_delays.neg().div(tdamp)))
        # k = self.x_cur_group[:,0].reshape((self.group_size,1))
        # s0 = self.x_cur_group[:,1].reshape((self.group_size,1))

        ## K0, xi, l, beta, tau
        K0 = self.x_cur_group[:,0].reshape((self.group_size,1))
        s = K0.sub(torch.sqrt(K0.pow(2).sub(self.input_Qr2.pow(2))))
        xi = self.x_cur_group[:,1].reshape((self.group_size,1))
        frac1 = (s.mul(xi)).pow(2).add(1)
        l = self.x_cur_group[:,2].reshape((self.group_size,1))
        frac2 = torch.sin(torch.sqrt(frac1).div(xi).mul(l).mul(math.pi)).pow(2)
        beta = self.x_cur_group[:,3].reshape((self.group_size,1))
        tau = self.x_cur_group[:,4].reshape((self.group_size,1))
        frac3 = torch.exp(self.input_delays.neg().div(tau)).mul(beta).add(1).sub(beta)
        I=frac2.mul(frac3).div(frac1)
        I0 = I[:,0].reshape((self.group_size,1))
        ratio = I.sub(I0).div(I0)
        Loss = torch.sum(self.validation_data.sub(ratio).pow(2),1,keepdim = True)
        where_are_nan = torch.isnan(Loss)
        source_inf = torch.full((where_are_nan.sum(),),float('inf')).double()
        Loss.masked_scatter_(where_are_nan,source_inf)
        return Loss
    def GD(self,x):
        self.x_cur_group.requires_grad_()
        loss_list_cur = self.Lossfunction()
        loss_list_cur.backward(torch.ones_like(loss_list_cur))        
        with torch.no_grad():     
            pre_process_grad_with_nan = self.x_cur_group.grad
            mask2 = torch.isnan(pre_process_grad_with_nan)
            source = torch.ones((mask2.sum(),)).double()
            no_nan_grad = pre_process_grad_with_nan.masked_scatter_(mask2,source)   
            self.v_cur_group = self.v_cur_group.mul(0.9).sub(no_nan_grad)
            self.x_cur_group = self.x_cur_group.add(self.v_cur_group.mul(0.1/x))
            self.x_cur_group = torch.min(self.x_cur_group,self.x_up_group)
            self.x_cur_group = torch.max(self.x_cur_group,self.x_down_group)
            self.v_cur_group = torch.min(self.v_cur_group,self.v_up_group)
            self.v_cur_group = torch.max(self.v_cur_group,self.v_down_group)
    def evolution(self):
        with torch.no_grad():
            r1,r2 = torch.rand(2).double()
            self.v_cur_group = self.v_cur_group.mul(self.weight).add(self.x_history_best_group.sub(self.x_cur_group).mul(self.c1*r1)).add(self.x_history_best_total.sub(self.x_cur_group).mul(self.c2*r2))
            self.x_cur_group = self.x_cur_group.add(self.v_cur_group)
            if torch.ge(torch.rand(1),0.8):
                print('variation happen')
                x_cur_variation_group = torch.rand_like(self.x_cur_group).mul(2)
                self.x_cur_group = x_cur_variation_group.mul(self.x_cur_group)
            self.x_cur_group = torch.min(self.x_cur_group,self.x_up_group)
            self.x_cur_group = torch.max(self.x_cur_group,self.x_down_group)
            self.v_cur_group = torch.min(self.v_cur_group,self.v_up_group)
            self.v_cur_group = torch.max(self.v_cur_group,self.v_down_group)
    
    def evaluate(self):
        with torch.no_grad():
            loss_list_cur  = self.Lossfunction()
            mask1 = torch.lt(loss_list_cur,self.loss_history_best_group)
            print('efficiency:',mask1.sum().item()*100/self.group_size,'%')
            temp = torch.masked_select(self.x_cur_group,mask1)
            self.x_history_best_group = self.x_history_best_group.masked_scatter(mask1,temp)
            self.loss_history_best_group = torch.min(self.loss_history_best_group,loss_list_cur)
            loss_total_cur, index = torch.min(self.loss_history_best_group,0)
            if torch.lt(loss_total_cur,self.loss_history_best_total):
                self.x_history_best_total = self.x_history_best_group[index,:]
                self.loss_history_best_total = loss_total_cur
                print(self.name)
                print(self.loss_history_best_total.item())
                print(self.x_history_best_total.tolist())
                with open('x_history_best_total.txt', 'a') as file:
                    file.write(str(self.loss_history_best_total.item())+'\n')
                    file.write(str(self.x_history_best_total.tolist())+'\n')

                

    

particlegroup1 = ParticleGroup(800000,1,0.5,1.5,'Group1')
particlegroup2 = ParticleGroup(800000,1.0,0.5,2.0,'Group2')
def work1():
    for i in range(1000):
        print('epoch',i)
        particlegroup1.evolution()
        print('evolution finished')
        particlegroup1.GD(i+1)
        print('GD1 finished')
        particlegroup1.GD(i+1)
        print('GD2 finished')
        particlegroup1.evaluate()
def work2():
    for i in range(1000):
        print('epoch',i)
        particlegroup2.evolution()
        print('evolution finished')
        particlegroup2.GD(i+1)
        print('GD1 finished')
        particlegroup2.GD(i+1)
        print('GD2 finished')
        particlegroup2.evaluate()

if __name__ == '__main__':
    p1 = mp.Process(target=work1)
    p2 = mp.Process(target=work2)
    p1.start()
    p2.start()
    p1.join()
    p2.join()








