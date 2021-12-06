import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as scsl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import utilities.util_kPCA as util

############### DATA READING AND NORMALIZATION ###############
saving=True
nnear=100
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

############### Kernel-PCA PART ###############
print("Computing Distance...")
Distances=util.Distance(X)
#Average 5th Neighbor distance
print("Computing sigma...")
sigma2=np.average(np.sqrt(np.sort(Distances,axis=1)[:,nnear]))**2
print("sigma2",sigma2)
print("Computing Kernel...")
Ker=  1-np.exp(-0.5*Distances/sigma2) # Distances # 
#Gram Matrix
print("Centering...")
G=util.Double_Centering(Ker)
#Solve Eigenvalue problem
print("Diagonalization...")
eps, Y = scsl.eigsh(-G,k=D)
print("eps_",eps)
Y=Y.T*np.reshape(np.sqrt(-eps),newshape=(D,1))
print("first eigenvec",Y[0,:5])
eps=-eps/N

############### DATA ANALYSIS ###############
#Partial Weights of Cs
Total_Geps=np.sum(eps)
Partial_Geps=0.0
print("********** PCs weights **********")
for i in range(D):
    print(f"index: {i} - weight: {eps[i]/Total_Geps}")
    if(Partial_Geps<0.95):
        i_max=i
        Partial_Geps+=eps[i]/Total_Geps
print(f"With kPCA adopting a Gaussian Kernel with sigma={np.sqrt(sigma2)}")
print(f"We need {i_max} component to explain {100*Partial_Geps}% of the spectrum")
print(f"The first 2 componets account only for {100*np.sum(eps[:2])/Total_Geps}% of the Spectrum")
print("********** **********")


plt.figure()
plt.scatter(range(len(eps)),eps)
plt.yscale("log")
plt.ylabel("$ \epsilon_n $")
plt.xlabel("n")
if(saving):    plt.savefig("Gaussian_kPCA_spectrum_logy")
plt.show()
plt.close()

Ymax=np.amax(Y,axis=1)+0.1
Ymin=np.amin(Y,axis=1)-0.1
#N_Color=len(ToColor)

plt.figure("PCA")
j=0
for j,c in enumerate(labels_list):
    mask=(labels==c)
    plt.scatter(Y[0,:][mask],Y[1,:][mask],label=f"label = {c}",alpha=0.7,marker="o",color=f"{colors[j]}")
    plt.xlim(Ymin[0],Ymax[0])
    plt.ylim(Ymin[1],Ymax[1])
    j+=1
plt.legend()
if(saving): plt.savefig("Labels_Togheter_kPCA")
plt.show()

fig, axs = plt.subplots(Nlabels,figsize=(5,5*Nlabels))
fig.suptitle(f"Subplots for Labels")
j=0
for j,c in enumerate(labels_list):
    mask=(labels==c)
    axs[j].scatter(Y[0,:][mask],Y[1,:][mask],label=f"label = {c}",alpha=0.7,color=f"{colors[j]}")
    axs[j].set_xlim(Ymin[0],Ymax[0])
    axs[j].set_ylim(Ymin[1],Ymax[1])
    axs[j].legend()
    j+=1
if(saving): plt.savefig("Labels_kPCA")
plt.show()

acc_list=[]
MI_list=[]

N_learn=9000
for Nreg in range(2,D+1):
    print(f"Doing {Nreg} components...")
    LR = LogisticRegression(max_iter=500).fit(Y[:Nreg,:N_learn].T, labels[:N_learn])
    prediction=LR.predict(Y[:Nreg,N_learn:].T)
    acc=accuracy_score(prediction,labels[N_learn:])
    print(f"For {Nreg} Kernel Principal Components:")
    print(f"... the accuracy of the regression is {acc} ")
    MI=util.Mutual_Information(prediction,labels[N_learn:],labels_list)
    print(f"... the Mutual Information is {MI}")
    print("********************************************")
    acc_list.append(acc)
    MI_list.append(MI)

fig, ax1 = plt.subplots() 
fig.suptitle("Accuracy & Mutual Information")
ax1.set_xlabel('Number or kernel PC') 
ax1.set_ylabel('Accuracy', color = 'red') 
ax1.plot(range(2,D+1), acc_list, color = 'red') 
ax1.tick_params(axis ='y', labelcolor = 'red') 
ax2 = ax1.twinx() 
ax2.set_ylabel('Mutual Information', color = 'blue') 
ax2.plot(range(2,D+1),MI_list, color = 'blue') 
ax2.tick_params(axis ='y', labelcolor = 'blue') 
if(saving): plt.savefig("PCA_ACCvsMI")
plt.show()

np.savetxt("sigma2.txt",[sigma2])