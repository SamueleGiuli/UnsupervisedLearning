import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as scsl
import util_kPCA as util
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

############### DATA READING AND NORMALIZATION ###############
saving=False
datapath="./data/data_kPCA.txt"
labelspath="./data/labels_kPCA.txt"
colors=np.array(["red","blue","green","orange","black"])
#Getting data and labels
X = np.loadtxt(datapath,unpack=True)[:,:10]
labels = np.loadtxt(labelspath,dtype=str,unpack=True)[:10]
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
sigma2=np.average(np.sort(Distances,axis=1)[:,4])
print("sigma2",sigma2)
print("Computing Kernel...")
Ker=Distances # np.exp(-0.5*Distances/sigma2)
#Gram Matrix
print("Centering...")
G=util.Double_Centering(Ker)
G=np.matmul(X.T,X)
#Solve Eigenvalue problem
print("Diagonalization...")
eps, Y = np.linalg.eigh(G)
eps=eps[::-1]
eps=eps/N
Y=Y[:,::-1].T
print("first eigenvec",Y[0,:5], "norm:",np.linalg.norm(Y[0,:]))

#Partial Weights of Cs
Total_Geps=np.sum(eps)
print("first eigenvec2",Y[0,:5]/Total_Geps)
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
plt.show()
if(saving):    plt.savefig("Gaussian_kPCA_spectrum_logy")
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

N_learn=9000
for Nreg in range(2,D+1):
    LR = LogisticRegression(max_iter=500).fit(Y[:Nreg,:N_learn], labels[:N_learn])
    prediction=LR.predict(Y[:Nreg,N_learn:])
    acc=accuracy_score(prediction,labels[N_learn:])
    print(f"For {Nreg} Kernel Principal Components the accuracy of the regression is {acc} ")