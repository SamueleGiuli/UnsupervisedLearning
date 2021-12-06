import numpy as np
import matplotlib.pyplot as plt

datapath="./winequality-red.csv"
saving=False

# Rows are data points and columns are features
X = np.loadtxt(datapath, delimiter=";", skiprows=1, usecols=range(0,11),unpack=True)


D,N=X.shape
print(f"N:{N}, - D:{D}")

#Normalization
means=np.average(X,axis=1)
stdX=np.std(X,axis=1)
X = ( X-np.reshape(means,newshape=(D,1)) )/np.reshape(stdX,newshape=(D,1))

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
print("********** **********")


plt.figure()
plt.scatter(range(len(eps)),eps)
#plt.yscale("log")
plt.ylabel("$ \epsilon_n $")
plt.xlabel("n")
plt.show()
plt.close()

Y=np.matmul(vec,X)

for f in ToColor:
    c=ToColor[f]
    plt.figure()
    plt.title(f)
    mask=(datacolor==c)
    plt.scatter(Y[0,:][mask],Y[1,:][mask],c=datacolor[mask],label=f,alpha=0.7)
    if(saving):
        plt.savefig(f"{whatcolor}_{f}")
    plt.close()


plt.figure()
for f in ToColor:
    c=ToColor[f]
    mask=(datacolor==c)
    plt.scatter(Y[0,:][mask],Y[1,:][mask],c=datacolor[mask],label=f,alpha=0.3)
    if(saving):
        plt.savefig(f"{whatcolor}_{f}")
plt.legend()
plt.show()
plt.close()

if(False):
    plt.figure()
    plt.title(whatcolor)
    plt.scatter(Y[0,:],Y[1,:],c=datacolor,alpha=0.3)
    plt.savefig(f"{whatcolor}_{f}")
    plt.bar(range(len(ToColor)), list(ToColor.values()),align='center')
    plt.xticks(range(len(ToColor)), list(ToColor.keys()))
    plt.show()
