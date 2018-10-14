import numpy as np
import os
from matplotlib.image import imread
import matplotlib.pyplot as plt
pathname = 'C:\\Users\\17154\\Documents\\GaTe_UED SLAC_to Lanxin Jia\\UED data_20180322\\D3\\scan9\\'
scans = [i+1 for i in range(9)]
nruns = len(scans)
Timezero = -8.319
pxiscale = 0.02
picname_inscan = []
for i in scans:
    str1 = 'run{:0>3d}'.format(i)
    str2 = '\\images-ANDOR1\\'
    pathtemp = pathname+str1+str2
    filesname_inrun = os.listdir(pathtemp)
    picname_inrun = []
    for j in filesname_inrun:
        if os.path.splitext(j)[1] == '.tif':
            picname_inrun.append(j)
    picname_inscan.append(picname_inrun)
picname_inscan = np.array(picname_inscan)
Alldelays = []
Fileinfo = picname_inscan[0]
for k in Fileinfo:
    namesplit1 = k.split('_')
    namesplit2 = namesplit1[1].split('-')
    Alldelays.append(-float(namesplit2[3]))
Alldelays = np.array(Alldelays)
DelaysM = np.unique(Alldelays)
Delays = 6.67*(DelaysM-Timezero)
nstep = Delays.size
nimstep = Fileinfo.size/nstep
a = np.diff(np.diff(Delays))
b = np.round(a)
c = np.mean(a)
d = np.round(c)
delind = np.where(b != d)[0]
if delind.size == 0:
    delind = Delays.size
if np.remainder(delind, 2) != 0:
    delind = delind-1
onepath = pathname+str1+str2+picname_inscan[-1, -1]
RefIm = imread(onepath)
print(RefIm.shape)
Alignment = input('align to previous points?(y/n)')
if Alignment == 'n':
    NoAlignRef = 2
    accuracy = 1
    method = 'nearest'
    color = 'jet'
    refrr = np.zeros((NoAlignRef, 4))
    for l in range(NoAlignRef):
        plt.figure()
        plt.imshow(RefIm, cmap='jet')
        plt.show()
        refr = input('left top/right bottom point position?x1,y1,x2,y2:')
        refr = list(map(float, refr.split(',')))
        refrr[l, 0] = np.round(refr[0])
        refrr[l, 1] = np.round(refr[1])
        refrr[l, 2] = np.round(refr[2]-refr[0])
        refrr[l, 3] = np.round(refr[3]-refr[1])
    np.savetxt('RefROIs.csv', refrr)
if Alignment == 'y':
    refrr = np.loadtxt(open('RefRoIs.csv', 'rb'))
Oldpeaks = input('analyze old peaks?(y/n)')
if Oldpeaks == 'n':
    peakpairs_X = np.array([[0, 0]])
    peakpairs_Y = np.array([[0, 0]])
    center_X = np.array([])
    center_Y = np.array([])
    while True:
        Newpeakpairs = input('new peak pairs?(y/n)')
        if Newpeakpairs == 'y':
            plt.figure()
            plt.imshow(RefIm, cmap='jet')
            plt.show()
            peakpair = input('peak pair loacation?x1,y1,x2,y2:')
            peakpairposition = list(map(float, peakpair.split(',')))
            X1 = np.round(peakpairposition[0]).astype(int)
            Y1 = np.round(peakpairposition[1]).astype(int)
            X2 = np.round(peakpairposition[2]).astype(int)
            Y2 = np.round(peakpairposition[3]).astype(int)
            SS = int(15)
            Xcom = np.zeros((1, 2))
            Ycom = np.zeros((1, 2))
            tempmatrix1 = np.arange(X1-SS, X1+SS+1).reshape((2*SS+1, 1))
            tempmatrix2 = np.sum(
                RefIm[Y1-SS-1:Y1+SS, X1-SS-1:X1+SS], axis=1, keepdims=True)
            Xtemp1 = np.sum(np.multiply(tempmatrix1, tempmatrix2))
            Xtemp2 = np.sum(tempmatrix2)
            Xcom1 = Xtemp1/Xtemp2
            tempmatrix3 = np.arange(Y1-SS, Y1+SS+1).reshape((2*SS+1, 1))
            tempmatrix4 = np.sum(
                RefIm[Y1-SS-1:Y1+SS, X1-SS-1:X1+SS], axis=0, keepdims=True).reshape((2*SS+1, 1))
            Ytemp1 = np.sum(np.multiply(tempmatrix3, tempmatrix4))
            Ytemp2 = np.sum(tempmatrix4)
            Ycom1 = Ytemp1/Ytemp2
            Xcom[0, 0] = np.round(Xcom1)
            Ycom[0, 0] = np.round(Ycom1)
            tempmatrix1 = np.arange(X2-SS, X2+SS+1).reshape((2*SS+1, 1))
            tempmatrix2 = np.sum(
                RefIm[Y2-SS-1:Y2+SS, X2-SS-1:X2+SS], axis=1, keepdims=True)
            Xtemp1 = np.sum(np.multiply(tempmatrix1, tempmatrix2))
            Xtemp2 = np.sum(tempmatrix2)
            Xcom1 = Xtemp1/Xtemp2
            tempmatrix3 = np.arange(Y2-SS, Y2+SS+1).reshape((2*SS+1, 1))
            tempmatrix4 = np.sum(
                RefIm[Y2-SS-1:Y2+SS, X2-SS-1:X2+SS], axis=0, keepdims=True).reshape((2*SS+1, 1))
            Ytemp1 = np.sum(np.multiply(tempmatrix3, tempmatrix4))
            Ytemp2 = np.sum(tempmatrix4)
            Ycom1 = Ytemp1/Ytemp2
            Xcom[0, 1] = np.round(Xcom1)
            Ycom[0, 1] = np.round(Ycom1)
            Xcen = np.mean(Xcom)
            Ycen = np.mean(Ycom)
            peakpairs_X = np.vstack((peakpairs_X, Xcom))
            peakpairs_Y = np.vstack((peakpairs_Y, Ycom))
            center_X = np.append(center_X, Xcen)
            center_Y = np.append(center_Y, Ycen)
        if Newpeakpairs == 'n':
            break
    X0 = np.mean(center_X)
    Y0 = np.mean(center_Y)
    peakpairs_X = np.split(peakpairs_X, [1, peakpairs_X.shape[0]], axis=0)[1]
    peakpairs_Y = np.split(peakpairs_Y, [1, peakpairs_Y.shape[0]], axis=0)[1]
    np.savetxt('Peaks_X.csv', peakpairs_X)
    np.savetxt('Peaks_Y.csv', peakpairs_Y)
    np.savetxt('center.csv', np.array([X0, Y0]))
if Oldpeaks == 'y':
    peakpairs_X = np.loadtxt('Peaks_X.csv')
    peakpairs_Y = np.loadtxt('Peaks_Y.csv')
    [X0, Y0] = np.loadtxt('Center.csv')


def analyze_runs(m):
    rSize1 = 20
    rSize2 = 20
    tempCamBG = np.zeros((Alldelays.size, 1))
    tempshifts = np.zeros((Alldelays.size, 2))
    tempImSum = np.zeros((Alldelays.size, 1))
    tempNormIm = np.zeros((Alldelays.size, RefIm.shape[0], RefIm.shape[1]))
    for n in range(Alldelays.size):
        namessplit1 = picname_inscan[m, n].split('_')
        namessplit2 = namessplit1[1].split('-')
        deli = -float(namessplit2[3])
        num = np.where(DelaysM == deli)[0][0]+1
        stri1 = 'run{:0>3d}'.format(i)
        stri2 = '\\images-ANDOR1\\'
        pathtempr = pathname+stri1+stri2+picname_inscan[m-1][n]
        currentImage = imread(pathtempr)
        tempCamBG[m-1][0] = np.mean(currentImage[:40][:40])
        BGIM = currentImage - tempCamBG[m-1][0]
        [AlIm, tempshifts(i,:)] = imalign2((RefIm,BGIm),refrr,accuracy,method,color)
