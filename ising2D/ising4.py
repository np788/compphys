import math, pylab, random
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

J = 1.0  
L = 20
nsweep = 100
#seed = input('random number seed -> ')
#random.seed(seed)
N = L**2
kT = np.arange(2.0, 2.265, 0.00625) #This means the array [ 0.1 0.2 0.3 ...... 4.9]
e = np.zeros(len(kT)) #len(kT) = 49
m = np.zeros(len(kT))
c = np.zeros(len(kT)) 
tc = 2.269 # critical temperature

# initial data
s = np.ones((L, L)) # This is a L x L matrix
E = 0.0
M = 0.0
for i in range(L):
    for j in range(L):
        E -= J*s[i,j]*(s[(i+1)%L,j]+s[i,(j+1)%L]) #periodic
        M += s[i,j]
tee = np.abs((kT - tc)/tc) #Since kT is a vector, tee is a vector too (with 49 terms)
em = []

# slowly warm up
for t in range(len(kT)): #this ranges over the temperatures, so t = 0,1,2,...,48
    # average nsweep sweeps
    for sweep in range(nsweep): #for each temperature, so sweep = 0,1,2,...,99. Sweep is like time, so that the system thermalizes
        print("Now on sweep ",t, sweep) 

        # sweep over all particles in lattice
        for i in range(L):
            for j in range(L):

                # compute energy required to flip spin
                dE = s[(i+1)%L,j]+s[(i-1)%L,j]+s[i,(j+1)%L]+s[i,(j-1)%L]
                dE *= 2.0*J*s[i,j]

                # Metropolis algorithm to see if we should accept trial
                if dE <= 0.0 or random.random() <= math.exp(-dE/kT[t]):
                    # accept trial: reverse spin; return dE and dM
                    s[i,j] *= -1
                    M += 2.0*s[i,j]
                    E += dE
                    
        # update running means and variances
        deltae = E-e[t]
        deltam = M-m[t]
        e[t] += deltae/(sweep+1)
        m[t] += deltam/(sweep+1)
        c[t] += deltae*(E-e[t])
        
    e[t] /= N
    m[t] /= N
    c[t] /= nsweep*N*kT[t]**2
    em.append(M)




#Plot
kT = tc - kT

line = 1.1034*kT**(1/8.0)
  
fig = plt.figure()
ax = plt.gca()
ax.scatter(kT ,m , c='blue', label= 'Magnetization', alpha=1)
ax.plot(kT ,line , c='green', label = '1/8 slope', alpha=1)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim([-10,1])
ax.set_ylim([-10,1])
plt.xlabel('Temperature ($J/k_B$)')
plt.ylabel('Magnetization ($J/\mu$)')
plt.title('Log-log plot of magnetization vs. temperature')
plt.grid()
plt.legend()
plt.show()


#Beta
logkT = np.log(kT)
logm = np.log(m)
slope, intercept = stats.linregress(logkT, logm)[0:2]
print()
print("Beta from least-squares fit: ", slope)







