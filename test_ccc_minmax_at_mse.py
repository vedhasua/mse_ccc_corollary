# (c) Vedhas Pandit
# The following script reaffirms the formulation presented in Section 4: Equations 45 and 45.
# That is, if \Upsilon(t) is defined to be 2t/(1+t^2), then
#     CCC_max = \Upsilon(1+abs(mse/2std(G)))=defined to be \Phi(abs(mse/2std(G))), 
#        obtained when errors are equally and positively proportional to the deviation of the gold standard value from the gold standard mean.
# and CCC_min = \Upsilon(1-abs(mse/2std(G)))=defined to be \phi(abs(mse/2std(G))), 
#        obtained when errors are equally and positively proportional to the deviation of the gold standard value from the gold standard mean.
# Using the script below, one can test the established identity using own gold standard sequence (GoldSeq) and/or own chosen value of MSE, and test against an error sequence of choice (ErrData) amounting to the same MSE. 

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

GoldSeq    =   Train_L[BegTime:EndTime,0][::Skip]           # Gold Standard = GoldSeq = Train_L[0 to 1500] in steps of 10

##### If MSE is small enough, specifically less than std(GoldSeq)^2, CCC is bound to be non-negative.
print("\n....\tStandard deviation of the gold standard sequence you gave me is {:.5f}.".format(np.std(GoldSeq,ddof=0)))
print("\tThus, for any MSE < {:.5f} (={:.5f}^2), it is impossible to get CCC<0, \n\tno matter how imaginative I am in generating the error sequence.".format(np.std(GoldSeq,ddof=0)**2,np.std(GoldSeq,ddof=0) ))
print("\tYou can try tweaking 'MSE' to cross-check this! :)")
MSE         =   np.pi/100 #0.00166 #
print("\n....\tYou have now set MSE = {:.5f}".format(MSE))

#### Compute the multiplier = abs(np.sqrt(MSE/np.std(GoldSeq,ddof=0))) as per Equations 30 and 41, to generate the optimised error sequence.
multiplier  =   abs(np.sqrt(MSE)/np.std(GoldSeq,ddof=0))
Err_optCCC  =   multiplier*(GoldSeq-np.mean(GoldSeq))
MSE2        =   np.sum(np.power(Err_optCCC,2))/Err_optCCC.size
print("\n....\tGiven your MSE and the gold standard sequence, I came up with error sequence that would maximise (and minimise) the CCC.")
print("\tVerify that the given MSE (={:.5f}) equals the squared sum of the optimised error sequence I just computed (={:.5f}).".format(MSE,MSE2))

######## Optimised prediction sequences for the given MSE and the given gold standard sequence, yeilding CCC = CCC_min and CCC_max. 
p_maxCCC = GoldSeq + Err_optCCC
p_minCCC = GoldSeq - Err_optCCC

cccMax=calc_scores(GoldSeq,p_maxCCC)[0]
cccMin=calc_scores(GoldSeq,p_minCCC)[0]
print("\n....\tI see that with the MSE and gold standard you gave me, ")
print("\tCCC (gold, pred_maximum) = {:.5f}".format(cccMax))
print("\tCCC (gold, pred_minimum) = {:.5f}".format(cccMin))

######## Any other error sequence with the same squared sum is bound to generate a prediction, with CCC that is between CCC_min and CCC_max.

print("\n....\tNow I am going to generate a random error sequence with the squared sum = given MSE.")
ErrData = np.random.uniform(0,0.2,size=GoldSeq.size)
ErrData = np.sqrt(MSE*ErrData.size/np.sum(np.power(ErrData,2)))*ErrData
MSE3 = np.sum(np.power(ErrData,2))/ErrData.size
print("\tVerify that the given MSE (={:.5f}) equals the squared sum for the error sequence I just generated (={:.5f}).".format(MSE,MSE3))
print("\tWith this error sequence, I can generate two prediction sequences for the given gold standard sequence, P=G+E and P=G-E.")
prediction1 = GoldSeq + ErrData
prediction2 = GoldSeq - ErrData
ccc1NonOpt=calc_scores(GoldSeq,prediction1)[0]
ccc2NonOpt=calc_scores(GoldSeq,prediction2)[0]
print("\n....\tCorrespondingly, ")
print("\tCCC (gold, prediction1) = {:.5f}".format(ccc1NonOpt))
print("\tCCC (gold, prediction2) = {:.5f}".format(ccc2NonOpt))
print("\tThese CCCs are bound to be between {:.5f} and {:.5f}.".format(cccMin, cccMax))

