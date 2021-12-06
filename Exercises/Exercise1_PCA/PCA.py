import numpy as np
import matplotlib.pyplot as plt

datapath="./AnuranCalls/Frogs_MFCCs.csv"
saving=False

arr_col=np.array(["black","blue","green","red","pink","purple","orange","brown","olive","cyan"])
arr_fam=np.array(["Bufonidae","Dendrobatidae","Hylidae","Leptodactylidae"])
arr_gen=np.array(["Adenomera","Ameerega","Dendropsophus","Hypsiboas","Leptodactylus","Osteocephalus","Rhinella","Scinax"])
arr_spc=np.array(["AdenomeraAndre","AdenomeraHylaedactylus","Ameeregatrivittata","HylaMinuta","HypsiboasCinerascens","HypsiboasCordobae","LeptodactylusFuscus","OsteocephalusOophagus","Rhinellagranulosa","ScinaxRuber"])

#Families
FamToColor=dict(zip(arr_fam,arr_col[:len(arr_fam)]))
GenToColor=dict(zip(arr_gen,arr_col[:len(arr_gen)]))
SpcToColor=dict(zip(arr_spc,arr_col[:len(arr_spc)]))

# Rows are data points and columns are features
X = np.loadtxt(datapath, delimiter=",", skiprows=1, usecols=range(0,22),unpack=True)
fam = np.loadtxt(datapath,dtype=str,delimiter=",", skiprows=1, usecols=22,unpack=True)
gen = np.loadtxt(datapath,dtype=str,delimiter=",", skiprows=1, usecols=23,unpack=True)
spc = np.loadtxt(datapath,dtype=str,delimiter=",", skiprows=1, usecols=24,unpack=True)


#CHANGE THIS
ToColor=FamToColor
whatcolor="Families"
whatdata =fam

datacolor = whatdata.copy()
for i,famdata in enumerate(whatdata):
    datacolor[i]=ToColor[famdata]

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
print(vec.shape)


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
plt.scatter(range(len(eps)),eps)
plt.yscale("log")
plt.ylabel("$ \epsilon_n $")
plt.xlabel("n")
plt.show()
if(saving):
    plt.savefig("VarX_spectrum_logy")
plt.close()

Y=np.matmul(vec,X)
Ymax=np.amax(Y,axis=1)+0.1
Ymin=np.amin(Y,axis=1)-0.1
N_Color=len(ToColor)


fig, axs = plt.subplots(N_Color,figsize=(5,5*N_Color))
fig.suptitle(f"Subplots for {whatcolor}")
j=0
for f in ToColor:
    c=ToColor[f]
    mask=(datacolor==c)
    axs[j].scatter(Y[0,:][mask],Y[1,:][mask],c=datacolor[mask],label=f,alpha=0.7)
    axs[j].set_xlim(Ymin[0],Ymax[0])
    axs[j].set_ylim(Ymin[1],Ymax[1])
    axs[j].legend()
    j+=1
plt.show()
    
#Saving Single Plots
for f in ToColor:
    c=ToColor[f]
    plt.figure()
    plt.title(f)
    mask=(datacolor==c)
    plt.scatter(Y[0,:][mask],Y[1,:][mask],c=datacolor[mask],label=f,alpha=0.7)
    plt.xlim(Ymin[0],Ymax[0])
    plt.ylim(Ymin[1],Ymax[1])
    if(saving):
        plt.savefig(f"{whatcolor}_{f}")
    plt.close()


plt.figure()
for f in ToColor:
    c=ToColor[f]
    mask=(datacolor==c)
    plt.xlim(Ymin[0],Ymax[0])
    plt.ylim(Ymin[1],Ymax[1])
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