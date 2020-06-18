# (c) Vedhas Pandit
# The following script reaffirms the formulation presented in Section 3: the Equation 15.
# That is, CCC = 1 - MSE(X,Y)/(MSE(X,Y)+2std(X,Y)) = 1/(1+MSE(X,Y)/2std(X,Y))
# Using the script below, one can test the established identity using own gold standard sequence (GoldSeq) and/or own set of error coefficients (p1r_Err). 

import numpy as np
import fnmatch
import os
from load_features     import load_all
from calc_scores       import calc_scores
from matplotlib import pyplot as plt
from glob import glob
from matplotlib import pyplot as plt
import random

### Directory initlization
path_labels = "labels/"

files_train = fnmatch.filter(os.listdir(path_labels), "Train*")  
# print(files_train)
# files_devel = fnmatch.filter(os.listdir(path_labels), "Devel*")

Train_L     = load_all( files_train, [ path_labels ] )  # Labels are not shifted, since shift=0 by default

#### Data Generation
Skip        =   10
BegTime     =   0
EndTime     =   1500
DurTime     =   EndTime-BegTime
TotTime     =   np.arange(BegTime,EndTime,1)[::Skip]

GoldSeq     =   Train_L[BegTime:EndTime,0][::Skip]           # Gold Standard = GoldSeq = Train_L[0 to 1500] in steps of 10
ErrData     =   0.1*np.power(10,np.linspace(0,1,DurTime))*0.2   
ErrData     =   ErrData[::Skip]                                 # Error coefficients, same length as of GoldSeq.

# Other example error coefficient sets to play with, or feel free to create your own: 
# ErrData   =   np.random.uniform(0,0.2,size=1500)
# ErrData   =   np.linspace(-0.2,0.4,DurTime) 
# ErrData   =  0.1*np.power(10,np.linspace(0,1,DurTime))*0.2   
# ErrData   =  0.1*np.cos(np.linspace(-0.2,0,DurTime)) #- 0.5 #+0.2
# ErrData   =   np.zeros_like(GoldSeq); ErrData[0:int(DurTime/2)] =   0.05*np.power(10,np.linspace(-0.4,0.4,int(DurTime/2)))

# Any random order of the given error coefficients
ErrRandomOrder  = random.sample(range(ErrData.shape[0]), ErrData.shape[0])         
p1r_Err         = ErrData[ErrRandomOrder]
p1r_Data        = GoldSeq+p1r_Err

#################################
# Compute and compare CCC values:
#################################

ccc1   = calc_scores(GoldSeq,p1r_Data)[0]
mse    = np.sum(np.power(p1r_Err,2))
stdXY  = np.cov(GoldSeq,p1r_Data,bias=1)[0][1]*(GoldSeq.size)
cccAlt = 1/(1+(0.5*mse/stdXY))
cccAlt2 = 1-(mse/(mse+2*stdXY))
print("CCC (gold, pred) = {:.5f}".format(ccc1))
print("CCC (gold, pred) = {:.5f}".format(cccAlt))
print("CCC (gold, pred) = {:.5f}".format(cccAlt2))
print("Check that the three lines above match exactly.")
