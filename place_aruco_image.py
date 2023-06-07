import cv2
import Marker_operation
import argparse




ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image for Aruco marker")
ap.add_argument("-l", "--markers", help="Path to the Marker file",type=str)
args = vars(ap.parse_args())

markers_path = ['Assets\Sample_Marker\Ha.jpg','Assets\Sample_Marker\HaHa.jpg','Assets\Sample_Marker\LMAO.jpg','Assets\Sample_Marker\XD.jpg']
image = cv2.imread(args["image"])
if args["markers"] is None:
    print("[Error] Marker file location is not provided")
    print("Using Default markers set")
else :
    inputed_markers_path = args["markers"]
    print(inputed_markers_path)
    markers_path = [marker.replace('\'','') for marker in inputed_markers_path.split(',')]
    print(markers_path)

markers = []
Marker_operation.straightenMarkers(markers,markers_path)

image_copy = image.copy()
output = Marker_operation.place_aruco_marker(image,markers)

w=800

cv2.imshow("Input",cv2.resize(image_copy,(w,int(w*output.shape[0]/output.shape[1])),interpolation=cv2.INTER_CUBIC))
cv2.imshow("Output",cv2.resize(output,(w,int(w*output.shape[0]/output.shape[1])),interpolation=cv2.INTER_CUBIC))
cv2.waitKey(0)
cv2.destroyAllWindows()