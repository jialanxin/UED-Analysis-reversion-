from numpy.fft import fft2
def imalign2(images, rect, accuracy=1 , method = 'linear',color = 'jet'):
    npoints = rect.shape[0]
    for o in range(npoints):
        refcrop = images[0][rect[o][1].astype(int)-1:rect[0][1].astype(int)+rect[0][3].astype(int)][rect[o][0].astype(int)-1:rect[0][0].astype(int)+rect[0][2].astype(int)]
        crop = images[1][rect[o][1].astype(int)-1:rect[0][1].astype(int)+rect[0][3].astype(int)][rect[o][0].astype(int)-1:rect[0][0].astype(int)+rect[0][2].astype(int)]
        fftrefcrop = fft2(refcrop)
        fftcrop  = fft(crop)
        