import numpy as np
import scipy.sparse.linalg as scsl
import util_various as util



def ISOMAP(X,ktype="Gauss",normz=True,nnear=10,ncomp=None,sigma=None):

    D,N=X.shape # BE SURE THE DIMENTIONS MEET WITH THIS DEFINITION   
    if(ncomp==None): ncomp=D
    #Normalization
    means=np.average(X,axis=1)
    stdX=np.std(X,axis=1)
    X = ( X-np.reshape(means,newshape=(D,1)) )/np.reshape(stdX,newshape=(D,1))

    Distances=util.Distance(X)
    if(sigma==None):
        sigma2=np.average(np.sqrt(np.sort(Distances,axis=1)[:,nnear]))**2
    else:
        sigma2=sigma**2
    maxDdist=np.max((np.sort(Distances,axis=1)[:,1]))
    if(maxDdist>sigma2):
        print("***** WARNING *****")
        print(f"The cut-off lenght ds:{sigma2}  is smaller than the maximum NN distance: {maxDdist}")
        print("*******************")
    Dgeo=util.FW_dist_sp(Distances,sigma2) #Floyd-Warshall algo for geodesic distance
    #Gram Matrix
    G=util.Double_Centering(Dgeo)
    eps, Y = scsl.eigsh(G,which="LA",k=ncomp)
    eps=eps[::-1]
    Y=Y[:,::-1]
    Y=Y.T*np.reshape(np.sqrt(eps),newshape=(ncomp,1))
    eps/=N

    return eps, Y