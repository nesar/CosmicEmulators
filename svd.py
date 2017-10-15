'''
SVD like K, W -> weights

Save W

K * W -> Prediction

Gotta save RowMean, std.dev etc

'''


import numpy as np
import scipy.linalg as SL
import matplotlib.pyplot as plt

#import SetPub
#SetPub.set_pub()

#def rescale_nM():
    #return (f - xmin)/(xmax - xmin) 

xlim1 = 10
xlim2 = 15
Mass = np.logspace(np.log10(10**xlim1), np.log10(10**xlim2), 500)[::10]

nsize = 5

hmf = np.loadtxt('Data/HMF_5Para.txt')
#OmegaM = hmf[:, 0]                ## OmegaM
#deltaH = hmf[:, 1] 
#X = rescale01( np.min(X), np.max(X), X)



y = hmf[:,5:].T # n(M)    -> all values 
yRowMean = np.zeros_like(y[:,0])

for i in range(y.shape[0]):
    yRowMean[i] = np.mean(y[i])

for i in range( y[0].shape[0] ):    
    y[:,i] = (y[:,i] - yRowMean)

stdy = np.std(y)
y = y/stdy

Pxx = y
U, s, Vh = SL.svd(Pxx, full_matrices=False)
assert np.allclose(Pxx, np.dot(U, np.dot(np.diag(s), Vh)))

NoEigenComp = 2

TruncU = U[:, :NoEigenComp]     #Truncation 
TruncS = s[:NoEigenComp]
TruncSq = np.diag(TruncS)
TruncVh = Vh[:NoEigenComp,:]

K = np.matmul(TruncU, TruncSq)/np.sqrt(NoEigenComp)
W1 = np.sqrt(NoEigenComp)*np.matmul(np.diag(1./TruncS), TruncU.T)
W = np.matmul(W1, y)

Pred = np.matmul(K,W)


for i in range(2, 11):
    plt.figure(i)
    plt.plot(Mass, y[:,i]*stdy + yRowMean, 'o', label = 'data')
    plt.plot(Mass, Pred[:,i]*stdy + yRowMean)

    plt.xscale('log')
#plt.yscale('log')

plt.show()

plt.figure(4343)
plt.plot(s, 'o-')
plt.xlim(-1,)
plt.ylim(-1,45)
plt.savefig('Plots/svd_fig1.png')

np.savetxt('Data/K_for2Eval.txt', K)   # Basis
np.savetxt('Data/W_for2Eval.txt', W)   # Weights

np.savetxt('Data/stdy.txt', [stdy])
np.savetxt('Data/yRowMean.txt',yRowMean)


#s[2:] = 0
#new_a = np.dot(U, np.dot(np.diag(s), Vh))
#print(new_a)
#plt.plot(new_a)
#plt.show()
#
#
#
#plt.figure(10)
#plt.plot()
