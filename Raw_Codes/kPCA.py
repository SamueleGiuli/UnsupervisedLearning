import numpy as np
import scipy.sparse.linalg as scsl
import util_various as util


#RETURNS ncomp eigenvals and eigenvecs of the Gram Matrix, if no ncomp is given it will return ncomp = D = number of features
#nnear-th nearest neighborg average distance square taken as typical lenght square if sigma is not given
def kPCA(X,ktype="Gauss",normz=True,nnear=10,ncomp=None,sigma=None):

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

    if(ktype=="Gauss"):
        Ker=  1-np.exp(-0.5*Distances/sigma2) 
    else:
        raise ValueError("Wrong kernel type inserted in kPCA")
        
    G=util.Double_Centering(Ker)
    eps, Y = scsl.eigsh(G,which="LA",k=ncomp)
    eps=eps[::-1]
    Y=Y[:,::-1]
    Y=Y.T*np.reshape(np.sqrt(eps),newshape=(ncomp,1))
    eps/=N

    return eps, Y