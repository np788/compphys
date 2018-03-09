import h5py
import numpy as np
import matplotlib.pyplot as plt
import cmath as cm
import time
import matplotlib.mlab as mlab
from pylab import plot, show, subplot, legend
from scipy.io import wavfile

#Define the relevant functions
def loaddata(filename):
    f = h5py.File(filename, "r")
    strain = f['strain/Strain'][...]
    t0 = f['strain/Strain'].attrs['Xstart']
    dt = f['strain/Strain'].attrs['Xspacing']
    t = t0 + dt * np.arange(strain.shape[0])
    f.close()
    return t, strain

def Hstep(f, f0, n):
    return 1/(1 + (f/f0)**(2*n))

def Hgauss(f, f0, sigma):
    return 1 - np.exp(-((f-f0)**2)/(2 * sigma**2))

def write_wavfile(filename,fs,data):
    d = np.int16(data/np.max(np.abs(data)) * 32767 * 0.9)
    wavfile.write(filename,int(fs), d)

def reqshift(data,fshift=100,sample_rate=4096):
    """Frequency shift the signal by constant
    """
    x = np.fft.rfft(data)
    T = len(data)/float(sample_rate)
    df = 1.0/T
    nbins = int(fshift/df)
    # print T,df,nbins,x.real.shape
    y = np.roll(x.real,nbins) + 1j*np.roll(x.imag,nbins)
    z = np.fft.irfft(y)
    return z


#Load the given data files and extract the relevant data
fn_H1 = 'H-H1_LOSC_4_V1-1126259446-32.hdf5'
fn_L1 = 'L-L1_LOSC_4_V1-1126259446-32.hdf5'
timeH, strainH = loaddata(fn_H1)
timeL, strainL = loaddata(fn_L1)



#Frequency and np.sizegth of data window
fs = 4096
dt = timeH[1] - timeH[0]
freqH = np.arange(0, fs, 1.0/32)

#The Fourier transforms of the strain data
fftValsH = np.fft.fft(strainH)  
fftValsL = np.fft.fft(strainL)
fftValsH2 = np.abs(fftValsH) * np.abs(fftValsH) * dt/np.size(fftValsH)
fftValsL2 = np.abs(fftValsL) * np.abs(fftValsL) * dt/np.size(fftValsL)
sqrtValsH = np.sqrt(fftValsH2)
sqrtValsL = np.sqrt(fftValsL2)
k = np.arange(np.size(fftValsH2))
fk = k/np.size(fftValsH2) * fs  



#The filters to apply to the signal
filter = (1 - Hstep(fk, 35, 8)) *Hstep(fk, 350, 8) * \
         Hgauss(fk, 40, 1) * Hgauss(fk, 60, 1) * \
         Hgauss(fk, 120, 1) * Hgauss(fk, 180, 1) * \
         Hgauss(fk, 334, 1) * Hgauss(fk, 35, 1) * \
         Hgauss(fk, 35.9, 1) * Hgauss(fk, 36.8, 1) * \
         Hgauss(fk, 331, 1) * Hgauss(fk, 510, 1) * \
         Hgauss(fk, 516, 1) * Hgauss(fk, 331, 1) * \
         Hgauss(fk, 501, 1)

fftValsH_filter = fftValsH * filter
fftValsL_filter = fftValsL * filter


filt_h1 = np.sqrt(np.abs(fftValsH_filter) * np.abs(fftValsH_filter) * dt/np.size(fftValsL))
filt_l1 = np.sqrt(np.abs(fftValsL_filter) * np.abs(fftValsL_filter) * dt/np.size(fftValsL))


#Calculate the inverse transforms of the filtered data
iFiltHhs = np.fft.ifft(fftValsH_filter)
iFiltLhs = np.fft.ifft(fftValsL_filter)

subplot(4,1,1)
plot(timeH,strainH, 'b-', label = "H strain")
plot(timeL,strainL, 'g-', label = "L strain")
plt.title("Time vs. strain")
plt.xlabel("Time (s)")
plt.ylabel("Strain h(t)")
legend(loc='upper left')


subplot(4,1,2)
plt.loglog(freqH, np.sqrt(fftValsH2),'b-', label = "H strain")
plt.loglog(freqH, np.sqrt(fftValsL2),'g-', label = "L strain")
plt.title("Periodogram")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Fourier values")
legend(loc='upper left')


subplot(4,1,3)
plt.loglog(freqH, np.array(np.abs(fftValsH_filter)),'b-', label = "H strain")
plt.loglog(freqH, np.array(np.abs(fftValsL_filter)),'g-', label = "L strain")
plt.title("Periodogram after filters")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Fourier values")
legend(loc='upper left')

subplot(4,1,4)
plot(timeH[2000:-2000], iFiltHhs[2000:-2000], 'b-')
plot(timeL[2000:-2000], iFiltLhs[2000:-2000], 'g-')
plt.title("Waveform of filtered data")
plt.xlabel("Time (s)")
plt.ylabel("Strain h(t)")
legend(loc='upper left')

show()


time = timeH
tevent = 1126259462.422         # Mon Sep 14 09:50:45 GMT 2015
deltat = 2.                     # seconds around the event

indxt = np.where((time >= tevent-deltat) & (time < tevent+deltat))


write_wavfile("H1whitenbp.wav", int(fs), iFiltHhs[indxt])
write_wavfile("L1whitenbp.wav", int(fs), iFiltLhs[indxt])

fshift = 400.
speedup = 1.
fss = int(float(fs)*float(speedup))

# shift frequency of the data
strain_H1_shifted = reqshift(iFiltHhs,fshift=fshift,sample_rate=fs)
strain_L1_shifted = reqshift(iFiltLhs,fshift=fshift,sample_rate=fs)

# write the files:
write_wavfile("H1shifted.wav",int(fs), strain_H1_shifted[indxt])
write_wavfile("L1shifted.wav",int(fs), strain_L1_shifted[indxt])
