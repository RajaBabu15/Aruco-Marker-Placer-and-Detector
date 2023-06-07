import numpy as np
import math

def isSquare(Quad):
    """
    Return slopes_of_side, internal_angles, isSquare:bool, 
    """
    d1=math.dist((Quad[0][0][0],Quad[0][0][1]),(Quad[2][0][0],Quad[2][0][1]))
    d2=math.dist((Quad[1][0][0],Quad[1][0][1]),(Quad[3][0][0],Quad[3][0][1]))
    slopes=[]
    thetas = []
    for i in range(4):
        dx=Quad[(i+1)%4][0][0]-Quad[i][0][0]
        dy=Quad[(i+1)%4][0][1]-Quad[i][0][1]
        if dx ==0:
            dx=0.0000001
        slopes.append(dy/dx)

    for i in range(4):
        p0=Quad[(i-1)%4][0]
        p1=Quad[i][0]
        p2=Quad[(i+1)%4][0]
        v0=np.array(p0)-np.array(p1)
        v1=np.array(p2)-np.array(p1)

        angle = np.arctan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
        thetas.append(np.degrees(angle))
    
    if (abs(1+slopes[0]*slopes[3])<0.1 and abs(1+slopes[0]*slopes[1])<0.11) and (abs(d1-d2)<(0.11*max(d1,d2))):
        return slopes,thetas,True
    else :
        return [],[],False