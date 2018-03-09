import numpy as np
import cmath as cm
import matplotlib.pyplot as plt
import time

#Define a slow DFT function
def dft(x):
    t = []
    N = np.size(x)
    for k in range(N):
        a = 0
        for n in range(N):
            a += x[n]*cm.exp(-2j*cm.pi*k*n*(1/N))
        t.append(a)
    return t

#A recursive Cooley-Tukey function
def fft(x):
    N = np.size(x)
    if N == 1:
        return x
    else:
        even = fft(x[::2])
        odd = fft(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([even + factor[:N / 2] * odd,
                               even + factor[N / 2:] * odd])

#An inverse DFT function    
def idft(t):
    x = []
    N = np.size(t)
    for n in range(N):
        a = 0
        for k in range(N):
            a += t[k]*cm.exp(2j*cm.pi*k*n*(1/N))
        a /= N
        x.append(a)
    return x

#These are the different n values that will be used for slow DFT and FFT
slowN = 12
fastN = 20

#These arrays will values needed to plot the performance times
Ndft = []
N = []
timeDFT = []
timeFFT = []
timenpFFT = []
nlogn = []
nsq = []

#This loop performs a series of slow DFTs and saves the computing time
#for each to an array. A prefactor of 1.8e-6 was applied for plotting purposes.
for i in range(slowN):
    x = np.random.random(2**i)
    Ndft.append(2**i)
    t0 = time.clock()
    dft(x)
    timeDFT.append(time.clock() - t0)
    nsq.append(1.8e-6 * (2**i)**2)

#This loop does the same, but for both FFT functions. Again, note the 2.4e-6
#prefactor for plotting. 
for i in range(fastN):
    x = np.random.random(2**i)
    N.append(2**i)
    t1 = time.clock()
    fft(x)
    timeFFT.append(time.clock() - t1)
    t2 = time.clock()
    np.fft.fft(x)
    timenpFFT.append(time.clock() - t2)
    nlogn.append(2.4e-6 * 2**i * np.log(2**i))


plt.plot(Ndft, timeDFT, 'ro', label = "DFT (slow)")
plt.plot(N, timeFFT, 'go', label = "My FFT")
plt.plot(N, timenpFFT, 'bo', label = "Numpy FFT")
plt.plot(N, nlogn, 'k-', label = "N log N")
plt.plot(Ndft, nsq, 'c-', label = "N^2")
plt.xlabel("Different n lengths used")
plt.ylabel("Time (s)")
plt.title("Log-log plot of time for different algorithms")
plt.legend(loc = 'upper left')
plt.xscale("log")
plt.yscale("log")
plt.show()

#Here, we test the inverse DFT function.
n = 5

x = np.random.random(2**n)
print("The original random numbers are ", x)
print("Feeding this into numpy's FFT and then running that output through my inverse DFT gives ", idft(np.fft.fft(x)))


