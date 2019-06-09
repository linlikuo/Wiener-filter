'''
Wiener filter and wiener deconvolution implemententation on webcam
'''

import os
import cv2
import matplotlib.pyplot as plt
from util import kernel, kernel2fft, cam
from wiener_filter import average_power_spectral_density, mean_power_spectrum_density
import matplotlib.animation as animation


def updatefig(*args):
    im.set_array(cam(cap, nu, lbd, mpsd, sigma))
    return im,

if __name__ == '__main__':
    n1, n2 = 240, 320
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, n1)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, n2)
    fig = plt.figure()

    train_folder = 'train'
    num_data = 5
    data_list = os.listdir(train_folder)
    train_data = [plt.imread(os.path.join(train_folder,d)) for d in data_list[:num_data]]

    apsd = average_power_spectral_density(train_data)
    #plt.figure()
    #showfft(apsd, apply_log=True, vmin=-10, vmax=5)
    mpsd, alpha, beta = mean_power_spectrum_density(apsd)
    print(alpha, beta)
    #plt.figure()
    #showfft(mpsd, apply_log=True, vmin=-10, vmax=5)
    
    sigma = 10/255
    tau = 2
    nu = (kernel('gaussian1',tau), kernel('gaussian2',tau))
    nu_combine = kernel('exponential',tau)
    lbd = kernel2fft(nu_combine, n1, n2)
    
    im = plt.imshow(cam(cap, nu, lbd, mpsd, sigma), animated=True)
    ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
    plt.show()