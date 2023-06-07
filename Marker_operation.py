import colorsys
import cv2
import numpy as np
import math
from utils import ARUCO_DICT,aruco_display
from environment import environ_setup
import Quadrilateral

def straightenMarkers(markers,markers_path):
    """
    @markers: original set of marker to which new markers has to append
    @markers_path: contains the set of the path of new markers
    
    """
    for path in markers_path:
        marker = cv2.imread(path)
        marker_gray=cv2.cvtColor(marker,cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(marker_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,-5)
        c,_ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(c,key=cv2.contourArea,reverse=True)
        blank=np.zeros(marker.shape,np.uint8)
        
        for i,contour in enumerate(c):
            if i==0:
                continue
            approxPloygon = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,closed=True),closed=True)
            
            M=cv2.moments(contour)
            if M['m00'] !=0.0 :
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])
            
            if len(approxPloygon) == 4:
                    cv2.drawContours(blank,[contour],-1,(255,0,0),2)
                    point_Mat = approxPloygon.copy().reshape(4,2)
                    topleft = point_Mat[0]
                    topright = point_Mat[1]
                    bottomright = point_Mat[2]
                    bottomleft = point_Mat[3]
                    
                    topleft = (int(topleft[0]),int(topleft[1]))
                    topright = (int(topright[0]),int(topright[1]))
                    bottomright = (int(bottomright[0]),int(bottomright[1]))
                    bottomleft = (int(bottomleft[0]),int(bottomleft[1]))

                    leftside_centerpoint = (int((topleft[0]+bottomleft[0])/2),int((topleft[1]+bottomleft[1])/2))
                    cv2.circle(blank,(x,y),5,(0,255,255),-1)
                    cv2.circle(blank,leftside_centerpoint,5,(0,0,255),-1)

                    center=(x,y)
                    point1=180
                    point2=math.degrees(math.atan2(leftside_centerpoint[0]-x,leftside_centerpoint[1]-y))
                    rotationAngle=(point1-point2)
                    
                    rotMat=cv2.getRotationMatrix2D(center=center,angle=rotationAngle,scale=1.0)
                    blank=cv2.warpAffine(blank,rotMat,(blank.shape[1],blank.shape[0]))
                    marker = cv2.warpAffine(marker,rotMat,(marker.shape[1],blank.shape[0]),cv2.BORDER_CONSTANT,borderValue=(255,255,255))

                    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT["DICT_5X5_100"])
                    arucoParams = cv2.aruco.DetectorParameters_create()
                    corners, ids, rejected = cv2.aruco.detectMarkers(marker, arucoDict, parameters=arucoParams)
                    detected_markers = aruco_display(corners, ids, rejected, marker)


                    idss=ids.reshape(1,-1)

                    markers.append((detected_markers,idss[0][0]))
                    break


def addMarker(image,Quad,markers):
    """
    Return the image with marker and the marker that was placed on the image

    @image: the initial image
    @Quad: contains the points the marker cordinate to place on the image
    @marker: marker that has to be placed on the image
    """
    polyArea = 0
    for i in range(4):
        polyArea=polyArea + Quad[i][0][0]*Quad[(i+1)%4][0][1]-Quad[(i+1)%4][0][0]*Quad[i][0][1]
    if polyArea >=0 or len(Quad) != 4:
        return image,None

    v1=(Quad[0][0][0],Quad[0][0][1])
    v2=(Quad[1][0][0],Quad[1][0][1])
    v3=(Quad[2][0][0],Quad[2][0][1])
    v4=(Quad[3][0][0],Quad[3][0][1])
    place_holder_center=((v1[0]+v3[0])//2,(v1[1]+v3[1])//2)

    
    img = image.copy()
    marker_center = (img.shape[1]//2,img.shape[0]//2)
    marker_white_bg = cv2.bitwise_not(np.zeros((img.shape[1],img.shape[0]),np.uint8))
    
    p1=(0,0) 
    p2=(img.shape[1],0)
    p3=(img.shape[1],img.shape[0])
    p4=(0,img.shape[0])
    
    marker_center = ((p1[0]+p3[0])//2,(p1[1]+p3[1])//2)
    scale=abs(0.9*math.dist(v1,v3)/math.dist(p1,p3))

    
    #Scaling Marker and its bg
    scaleMat = cv2.getRotationMatrix2D((marker_center[0]-35,marker_center[1]-35),90,scale)
    marker_white_bg = cv2.warpAffine(marker_white_bg,scaleMat,(img.shape[1],img.shape[0]))
    
    #Translating Marker and its bg
    tansMat=np.float64([[1,0,place_holder_center[0]-marker_center[0]],[0,1,place_holder_center[1]-marker_center[1]]])
    marker_white_bg=cv2.warpAffine(marker_white_bg,tansMat,(img.shape[1],img.shape[0]))
    
    #Rotating Marker and its bg
    vector0=np.array(((v1[0]+v2[0])//2,(v1[1]+v2[1])//2))-np.array(place_holder_center)
    vector1=np.array(((p1[0]+p2[0])//2+place_holder_center[0]-marker_center[0],(p1[1]+p2[1])//2+place_holder_center[1]-marker_center[1]))-np.array(place_holder_center)
    rotation_angle = np.degrees(np.arctan2(np.linalg.det([vector0,vector1]),np.dot(vector0,vector1)))
    rotMat = cv2.getRotationMatrix2D((place_holder_center[0].item(),place_holder_center[1].item()),rotation_angle,scale=0.9)
    marker_white_bg = cv2.warpAffine(marker_white_bg,rotMat,(img.shape[1],img.shape[0]))

    
    mean = cv2.mean(image,mask=marker_white_bg)
    bgr=(mean[0],mean[1],mean[2])
    h,s,v = convert_rgb_to_hsv(red=bgr[2],green=bgr[1],blue=bgr[0])
    marker = np.ones((100,100,3),np.uint8)
    i=None
    if (h>=45 and h<=75) and (s>=100 and s<=255) and (v>=100 and v<=255):
        i="Green"
        for j in range(4):
            if markers[j][1]==1:
                marker=markers[j][0]
    if (h>=15 and h<45) and (s>=100 and s<=255) and (v>=100 and v<=255):
        i="Orange"
        for j in range(4):
            if markers[j][1]==2:
                marker=markers[j][0]
    if (v<=15):
        i="Black"
        for j in range(4):
            if markers[j][1]==3:
                marker=markers[j][0]
    if (h>=15 and h<45) and (s>=15 and s<100) and (v>120):
        i="Pink-Peach"
        for j in range(4):
            if markers[j][1]==4:
                marker=markers[j][0]


    marker_copy =marker.copy()
    marker = cv2.resize(marker,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_LINEAR)
    scaleMat = cv2.getRotationMatrix2D((marker_center),0,scale)
    img2=cv2.warpAffine(marker,scaleMat,(img.shape[1],img.shape[0]))
    translated=cv2.warpAffine(img2,tansMat,(img.shape[1],img.shape[0]))
    rotated = cv2.warpAffine(translated,rotMat,(img.shape[1],img.shape[0]))
    
    pad = 13
    cont = np.array( [ [v1[0]-pad,v1[1]-pad], [v2[0]-pad,v2[1]+pad], [v3[0]+pad,v3[1]+pad], [v4[0]+pad,v4[1]-pad] ] )
    cv2.drawContours(image, [cont], -1, color=(0,0,0), thickness=cv2.FILLED)
    image = cv2.bitwise_xor(image,rotated)
    return image,marker_copy


def convert_rgb_to_hsv(red,green,blue):
    red_percentage= red / float(255)
    green_percentage= green/ float(255)
    blue_percentage=blue / float(255)
    color_hsv_percentage=colorsys.rgb_to_hsv(red_percentage, green_percentage, blue_percentage) 
    color_h=round(180*color_hsv_percentage[0])
    color_s=round(255*color_hsv_percentage[1])
    color_v=round(255*color_hsv_percentage[2])
    color_hsv=(color_h, color_s, color_v)

    return color_hsv

def place_aruco_marker(image,markers):
    _,_,img=environ_setup(image)
    contours,_ =cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for i,contour in enumerate(contours):
        if i==0:
            continue
        
        approxPloygon = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,closed=True),closed=True)
        if len(approxPloygon) == 4:
            _,_,isSquared= Quadrilateral.isSquare(approxPloygon.copy())
            
            if isSquared:
                image,_=addMarker(image,approxPloygon,markers)
    return image

