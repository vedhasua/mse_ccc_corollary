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
print(files_train)
#files_devel = fnmatch.filter(os.listdir(path_labels), "Devel*")

Train_L     = load_all( files_train, [ path_labels ] )  # Labels are not shifted, since shift=0 by default

#### Data Generation
Skip        =   10
BegTime     =   0
EndTime     =   1500
DurTime     =   EndTime-BegTime
TotTime     =   np.arange(BegTime,EndTime,1)[::Skip]

p0Data      =   Train_L[BegTime:EndTime,0][::Skip]              # Gold Standard = p0Data = Train_L[0 to 1500] in steps of 10
#ErrData     =   np.random.uniform(0,0.2,size=1500)
ErrData     =   0.1*np.power(10,np.linspace(0,1,DurTime))*0.2   
ErrData     =   ErrData[::Skip]                                 # Error coefficients, same length as of p0Data.

# Other example error coefficient sets to play with, or create feel free to create your own: 
# np.linspace(-0.2,0.4,DurTime) # 0.1*np.cos(np.linspace(-0.2,0,DurTime)) #- 0.5 #+0.2
# ErrData     =   np.zeros_like(p0Data)
# ErrData[0:int(DurTime/2)] =   0.05*np.power(10,np.linspace(-0.4,0.4,int(DurTime/2)))

ErrSort     =   np.argsort(-ErrData)    # sort errors in a descending order and note the corresponding indices, e.g., [15, 53, 5... ]
DatSort     =   np.argsort(-p0Data)     # sort gold-standard in a descending order and note the corresponding indices, e.g., [45, 89, 29.. ]

# Formulation1 (Equation 89 + section 9.1 of the paper) dictates that: 
#    For the prediction sequence that maximises CCC, given set{E}:
#       * Errors should be sorted in the same order as of the order in the gold standard, 
#                       and are added to the gold standard to generate the prediction sequence.
#       * Reasoning for the claim: Cf. Section 9.1) which states, [Gold standard=G=Y], [Prediction=P=X] ,   [D=X-Y].
#       * Consequently,                                         : [Error=E=P-G],          implying          [P=G+E].

p1Err           = np.zeros_like(p0Data)
p1Err[DatSort]  = ErrData[ErrSort]       # p1Error = sequence of error coefficients sorted in a descending order
p1Data          = p0Data+p1Err           # p1Data = p0Data+p1Error = corresponding prediction sequence maximising CCC

ErrRandomOrder  = random.sample(range(p1Err.shape[0]), p1Err.shape[0])         
p1r_Err           = np.zeros_like(p0Data)           # Try any other sequence by reordering of error coefficients, 
                                                    # but the resulting prediction sequence is bound to result in a lower CCC.  
p1r_Err[DatSort] = ErrData[ErrRandomOrder]
p1r_Data         = p0Data+p1r_Err

# Formulation2 (Equation 90 + section 9.2 of the paper) dictates that:
#    For the prediction sequence that maximises CCC, given set{E}:
#       * Errors should be sorted in the opposite order as of the order in the gold standard, 
#                       and are subtracted from the gold standard to generate the prediction sequence.
#       * Reasoning for the claim: Cf. Section 9.2 which states, [Gold standard=G=X], [Prediction=P=Y],    [D=X-Y].
#       * Consequently,                                        : [Error=E=G-P],          implying          [P=G-E].

p2Err           = np.zeros_like(p0Data)   
p2Err[DatSort]  = ErrData[ErrSort[::-1]] # p2Error = sequence of error coefficients sorted in a ascending order
p2Data          = p0Data-p2Err           # p2Data = p0Data-p2Error = corresponding prediction sequence maximising CCC.

# Minimum CCC achievable for the two formulations (Equation 99 + section 9.1, and Equation 100 + section 9.2 of the paper)
p1DataNeg     = p0Data+p2Err
p2DataNeg     = p0Data-p1Err

ErrRandomOrder  = random.sample(range(p2Err.shape[0]), p2Err.shape[0])         
p2r_Err           = np.zeros_like(p0Data)           # Try any other sequence by reordering of the error coefficients, 
                                                    # but the resulting prediction sequence is bound to result in a lower CCC.
p2r_Err[DatSort] = ErrData[ErrRandomOrder]
p2r_Data         = p0Data-p2r_Err

###########################
# The moment of truth! 
###########################
# Compute and compare CCC values:

ccc1=calc_scores(p0Data,p1Data)[0]
ccc2=calc_scores(p0Data,p2Data)[0]
print("CCC (gold, pred1_maximum) = {0:.5f}".format(ccc1))
print("CCC (gold, pred2_maximum) = {0:.5f}".format(ccc2))

ccc1r=calc_scores(p0Data,p1r_Data)[0]
ccc2r=calc_scores(p0Data,p2r_Data)[0]
print("CCC (gold, pred1_random) = {0:.5f}".format(ccc1r))
print("CCC (gold, pred2_random) = {0:.5f}".format(ccc2r))

# Generate plots/visualisations for the error, the prediction and the gold standard sequences
from pylab import subplot
fig, ax = plt.subplots(2, 2,sharey='row',sharex=True)
# Subplot(0,0)
ax[0][0].plot(TotTime,p0Data, 'g',label='Gold Standard')
ax[0][0].plot(TotTime,p1Data, 'r',label=' Prediction 1')
ax[0][0].text(0, 0.4, "CCC="+"{0:.3f}".format(ccc1), fontsize=10)
ax[0][0].set_ylabel('Arousal Level\n(Annotations)')
# Subplot(0,1)
ax[0][1].plot(TotTime,p0Data, 'g',label='Gold Standard')
ax[0][1].plot(TotTime,p2Data, 'b',label=' Prediction 2')
ax[0][1].text(0, 0.4, "CCC="+"{0:.3f}".format(ccc2), fontsize=10)
# Subplot(1,0)
ax[1][0].scatter(TotTime,p1Err[DatSort[::-1]],0.5, color='k', label='Errors Sorted (Ascending)')
ax[1][0].plot(TotTime,p1Data-p2Data, 'm',linewidth=1.1,label='Prediction 1 - Prediction 2')         # you might want to comment out this, 
                                                                                                    # if error is too small in comparison
ax[1][0].set_ylabel('Arousal Level\n(Errors)')
ax[1][0].set_xlabel('Instance number')
# Subplot(1,1)
ax[1][1].scatter(TotTime,p1Err[DatSort[::-1]],0.5, color='k', label='Errors Sorted (Ascending)')
ax[1][1].plot(TotTime,p1Data-p0Data, 'r',linewidth=1.1,label='Error Sequence 1')
ax[1][1].plot(TotTime,p0Data-p2Data, 'b',linewidth=1.1,label='Error Sequence 2')
ax[1][1].set_xlabel('Instance number')
# set figure size, legends, axis limits
ax[0][0].set_ylim([-0.25,0.5])                  # YOU MIGHT WANT TO COMMENT THIS LINE OUT/SET ylims differently if different ErrData is used.
ax[1][0].set_ylim([0,0.45])                     # YOU MIGHT WANT TO COMMENT THIS LINE OUT/SET ylims differently if different ErrData is used. :)
ax[0][0].legend(loc='lower right')
ax[0][1].legend()
ax[1][0].legend()
ax[1][1].legend()
fig.set_size_inches(7.2, 5.4)
'''
size = fig.get_size_inches()*fig.dpi
print(size,fig.get_size_inches(),fig.dpi)
'''
plt.tight_layout()

plt.show()
