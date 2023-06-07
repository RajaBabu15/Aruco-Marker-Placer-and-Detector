'''
environment.py nomalize the image 
'''

import cv2

def environ_setup(image):
    """
    environ_set take the 'image' as input

    and return image,gray,final_tansformed_image
    """

    image_copy=image.copy()

    #Converting Image into GrayScale
    gray = cv2.cvtColor(image_copy,cv2.COLOR_BGR2GRAY)
    #Setting Up Environment of image
    bilateralFilter = cv2.bilateralFilter(gray,d=5,sigmaColor=10,sigmaSpace=11)
    adaptiveThreshold = cv2.adaptiveThreshold(src=bilateralFilter,maxValue=255,adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,thresholdType=cv2.THRESH_BINARY,blockSize=13,C=4)
    dilated = cv2.dilate(adaptiveThreshold,kernel=(7,7),iterations=1)
    eroded = cv2.erode(dilated,kernel=(9,9),iterations=3)

    return image,gray,eroded