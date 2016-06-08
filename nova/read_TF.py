import numpy as np
import pylab as pl

config = 'SN'


file = '../Data/'+config+'_TFcoil.txt'
nL = sum(1 for line in open(file))-1  # file data length
Rin,Zin,Rout,Zout = np.zeros(nL),np.zeros(nL),np.zeros(nL),np.zeros(nL)
with open(file,'r') as f:
    f.readline()  # header
    for i,line in enumerate(f):
        line = line.split('\t')
        Rin[i] = float(line[0])
        Zin[i] = float(line[1])
        Rout[i] = float(line[2])
        Zout[i] = float(line[3])

pl.plot(Rin,Zin)
pl.plot(Rout,Zout)

