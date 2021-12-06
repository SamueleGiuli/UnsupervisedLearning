import numpy as np

def distance_types(dist_type):
    print("Choose distance by insering the relative index")
    print("Carthesian distance: 0")
    print("Local distance:     1")
    print("Other indexes will Raise an Error")
    dist_type=input("Insert index:")

def distance(X1,X2,dist_type,Ds=np.inf):
    if( len(X1) /= len(X2):
        raise ValueError("Input array in distance have different shapes")
    
    dist=np.dot(X1,X2)
    if(dist_type==0):
        return dist
    elif(dist_type==1):
        if(dist>Ds): dist=np.inf
        return dist
    else:
        raise ValueError("Wrong distance selection")