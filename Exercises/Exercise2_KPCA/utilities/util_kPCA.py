import numpy as np
from scipy.sparse.csgraph import floyd_warshall


def Distance(X):
    sp=X.shape
    N=sp[1]
    lin=np.reshape( np.sum(X**2,axis=0),newshape=(1,N))
    dist=-2*np.matmul(X.T,X) + lin +lin.T
    #dist=np.ze
    # ros((N,N))
    #for i in range(N):
    #    for j in range(i,N):
    #        d=np.sum( (X[:,i]-X[:,j])**2 )
    #        dist[i,j]=d
    #        dist[j,i]=d
    return dist

def Double_Centering(D):
    sp = D.shape
    if( (len(sp)!=2) or sp[0]!=sp[1] ):
        raise TypeError("Wrong shape on Double Centering input")
    N=sp[0]
    if( (D==None).any() ): raise ValueError("D has None inside!")
    if( (D==np.infty).any() ): raise ValueError("D has np.infty inside!")
    onesum=np.reshape(np.sum(D,axis=1),newshape=(N,1))/N
    twosum=np.reshape(np.sum(D,axis=0),newshape=(1,N))/N
    return -0.5*(D-onesum-twosum +np.sum(D)/N**2  )

#Floyd–Warshall Algorithm
def FW_dist(D,ds,pr=False):
    N=D.shape[0]
    Dgeo=np.where(D<ds , D, np.infty)
    cond=True
    loop=0
    #floyd algorithm
    while(cond):
        cond=False
        loop+=1
        if(pr): print("loop:",loop)
        for k in range(N):
            for i in range(N):
                for j in range(i,N):
                    d_old=Dgeo[i,j]
                    if(d_old==np.infty): 
                        if(loop==2): print("infi in ",i,j)
                    d_try=Dgeo[i,k]+Dgeo[k,j]
                    d_new=min(d_try,d_old)
                    Dgeo[i,j]=d_new
                    Dgeo[j,i]=d_new
        if( np.any(Dgeo==np.infty) ):
            print("some infty")
            print("N infty:",np.sum(Dgeo==np.infty))
            cond=True
    return Dgeo

def FW_dist_sp(D,ds,pr=False):
    N=D.shape[0]
    Dgeo=np.where(D<ds , D, np.infty)
    cond=True
    loop=0
    Dgeo = floyd_warshall(Dgeo)
    #floyd algorithm
    return Dgeo

def Mutual_Information(Prediction,Target,labels_list):
    if( Target.shape!=Prediction.shape):
        raise TypeError("Wrong shape on Mutual Information")
    Nlab =labels_list.shape[0]
    Ntest=Prediction.shape[0]
    P_P=np.zeros(Nlab)
    P_T=np.zeros(Nlab)
  #  print("Nlab - Ntest",Nlab,Ntest)
    for i,lab in enumerate(labels_list):
        P_P[i]=np.count_nonzero(Prediction==lab)
        P_T[i]=np.count_nonzero(Target==lab)
    P_P/=Ntest
    P_T/=Ntest
  #  print("P_P",P_P)
  #  print("P_T",P_T)
    MI=0.0
    for ip, labp in enumerate(labels_list):
        maskp=(Prediction==labp)    
        for it, labt in enumerate(labels_list):
            maskt=(Target==labt)
            Pboth=np.count_nonzero(maskp & maskt)/Ntest
            if(Pboth==0): continue
#            print("For",labp,labt," probab is ",Pboth)
            MI+=Pboth*np.log(Pboth/(P_P[ip]*P_T[it]))

    return MI