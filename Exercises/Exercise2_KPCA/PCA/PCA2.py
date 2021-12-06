import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import utilities.util_kPCA as util

############### DATA READING AND NORMALIZATION ###############
saving=True
datapath="../data/data_kPCA.txt"
labelspath="../data/labels_kPCA.txt"
colors=np.array(["red","blue","green","orange","black"])
#Getting data and labels
X = np.loadtxt(datapath,unpack=True)
labels = np.loadtxt(labelspath,dtype=str,unpack=True)
#Getting Parameters
D,N=X.shape
labels_list=np.unique(labels)
Nlabels=labels_list.shape[0]
print(f"N:{N}, - D:{D}")
print("N of labels:", Nlabels)
#Normalization
means=np.average(X,axis=1)
stdX=np.std(X,axis=1)
X = ( X-np.reshape(means,newshape=(D,1)) )/np.reshape(stdX,newshape=(D,1))
#
Ltest=1000
# Dividing into testing set of lenght Ltest
# and a learning set with the remaining
# TEST SET
X_T=X[:,:Ltest]
labels_T=labels[:Ltest]
# LEARNING SET
X_L=X[:,Ltest:]
labels_L=labels[Ltest:]

############### PCA PART ###############
#Variance
VarX=np.matmul(X,X.T)/N
#Solve Eigenvalue problem
eps, vec = np.linalg.eigh(VarX)
eps=eps[::-1]
vec=vec[:,::-1]
#Partial Weights of PCs
Total_Variance=np.sum(eps)
Partial_Variance=0.0
print("********** PCs weights **********")
for i in range(D):
    print(f"index: {i} - weight: {eps[i]/Total_Variance}")
    if(Partial_Variance<0.95):
        i_max=i
        Partial_Variance+=eps[i]/Total_Variance
print(f"We need {i_max} PC to explain {100*Partial_Variance}% of the variance")
print(f"The first 2 componets account only for {100*np.sum(eps[:2])/Total_Variance}% of the variance")
print("********** **********")


plt.figure()
plt.title("PCA spectrum")
plt.scatter(range(len(eps)),eps)
plt.yscale("log")
plt.ylabel("$ \epsilon_n $")
plt.xlabel("n")
if(saving):    plt.savefig("PCA_spectrum_logy")
plt.show()
plt.close()

Y_L=np.matmul(vec.T,X_L)
Y_T=np.matmul(vec.T,X_T)
Ymax=np.amax(Y_L,axis=1)+0.1
Ymin=np.amin(Y_L,axis=1)-0.1
#N_Color=len(ToColor)

print("first eigenvec",Y_L[0,:5],"norm:",np.linalg.norm(Y_L[0,:]))
print("first eigenvec",Y_L[0,:5]/np.linalg.norm(Y_L[0,:]))

plt.figure("PCA")
j=0
for j,c in enumerate(labels_list):
    mask=(labels_L==c)
    plt.scatter(Y_L[0,:][mask],Y_L[1,:][mask],label=f"label = {c}",alpha=0.7,marker="o",color=f"{colors[j]}")
    plt.xlim(Ymin[0],Ymax[0])
    plt.ylim(Ymin[1],Ymax[1])
    j+=1
plt.legend()
if(saving): plt.savefig("Labels_Togheter_PCA")
plt.show()

fig, axs = plt.subplots(Nlabels,figsize=(5,5*Nlabels))
fig.suptitle("Subplots for single Labels")
j=0
for j,c in enumerate(labels_list):
    mask=(labels_L==c)
    axs[j].scatter(Y_L[0,:][mask],Y_L[1,:][mask],label=f"label = {c}",alpha=0.7,color=f"{colors[j]}")
    axs[j].set_xlim(Ymin[0],Ymax[0])
    axs[j].set_ylim(Ymin[1],Ymax[1])
    axs[j].legend()
    j+=1
if(saving): plt.savefig("Labels_PCA")
plt.show()

acc_list=[]
MI_list=[]

for Nreg in range(2,D+1):
    LR = LogisticRegression(max_iter=500).fit(Y_L[:Nreg,:].T, labels_L[:])
    prediction=LR.predict(Y_T[:Nreg,:].T)
    acc=accuracy_score(prediction,labels_T)
    print(f"For {Nreg} Kernel Principal Components:")
    print(f"... the accuracy of the regression is {acc} ")
    MI=util.Mutual_Information(prediction,labels_T,labels_list)
    print(f"... the Mutual Information is {MI}")
    print("********************************************")
    acc_list.append(acc)
    MI_list.append(MI)
fig, ax1 = plt.subplots() 
fig.suptitle("Accuracy & Mutual Information")
ax1.set_xlabel('Number or PC') 
ax1.set_ylabel('Accuracy', color = 'red') 
ax1.plot(range(2,D+1), acc_list, color = 'red') 
ax1.tick_params(axis ='y', labelcolor = 'red') 
ax2 = ax1.twinx() 
ax2.set_ylabel('Mutual Information', color = 'blue') 
ax2.plot(range(2,D+1),MI_list, color = 'blue') 
ax2.tick_params(axis ='y', labelcolor = 'blue') 
if(saving): plt.savefig("PCA_ACCvsMI")
plt.show()