from random import Random
import cv2
import Marker_operation
from utils import ARUCO_DICT,aruco_display



image = cv2.imread("Assets\CVtask.jpg")
image_copy = image.copy()
markers=[]
markers_path = ['Assets\Sample_Marker\Ha.jpg','Assets\Sample_Marker\HaHa.jpg','Assets\Sample_Marker\LMAO.jpg','Assets\Sample_Marker\XD.jpg']
Marker_operation.straightenMarkers(markers,markers_path)


output = Marker_operation.place_aruco_marker(image,markers)

w=800

cv2.imshow("Input",cv2.resize(image_copy,(w,int(w*output.shape[0]/output.shape[1])),interpolation=cv2.INTER_CUBIC))
cv2.imshow("Output",cv2.resize(output,(w,int(w*output.shape[0]/output.shape[1])),interpolation=cv2.INTER_CUBIC))
cv2.waitKey(0)
cv2.destroyAllWindows()