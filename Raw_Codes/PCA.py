import numpy as np


#Return eigenvalues of the variance and the transformation matrix A such that Y=A*X in ascending order
def PCA(X,normz=True):

    D,N=X.shape # BE SURE THE DIMENTIONS MEET WITH THIS DEFINITION   

    if(normz):
        #Normalization
        means=np.average(X,axis=1)
        stdX=np.std(X,axis=1)
        X = ( X-np.reshape(means,newshape=(D,1)) )/np.reshape(stdX,newshape=(D,1))
    #Variance
    VarX=np.matmul(X,X.T)/N

    #Solve Eigenvalue problem
    eps, A = np.linalg.eigh(VarX)
    eps=eps[::-1]
    A=A[:,::-1]

    return eps, A