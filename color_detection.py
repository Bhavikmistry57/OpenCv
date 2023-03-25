import cv2
import numpy as np

def emptyfunc():
    pass

windowname = "color_Detector"
cv2.namedWindow(windowname)
################## WITH HSV ##################
# cv2.createTrackbar('upperHue', windowname, 153,180, emptyfunc)
# cv2.createTrackbar('upperSaturation', windowname, 255,255, emptyfunc)
# cv2.createTrackbar('upperValue', windowname, 255,255, emptyfunc)
# cv2.createTrackbar('lowerHue', windowname, 64,180, emptyfunc)
# cv2.createTrackbar('lowerSaturation', windowname, 72,255, emptyfunc)
# cv2.createTrackbar('lowerValue', windowname, 49,255, emptyfunc)

################### WITH BGR #################### 
cv2.createTrackbar('upperBlue', windowname, 0, 255, emptyfunc)
cv2.createTrackbar('upperGreen', windowname, 0, 255, emptyfunc)
cv2.createTrackbar('upperRed', windowname, 0, 255, emptyfunc)
cv2.createTrackbar('lowerBlue', windowname, 0, 255, emptyfunc)
cv2.createTrackbar('lowerGreen', windowname, 0, 255, emptyfunc)
cv2.createTrackbar('lowerRed', windowname, 0, 255, emptyfunc)

windowname2 = "color_Detector2"
cv2.namedWindow(windowname2)
cv2.createTrackbar('upperBlue2', windowname2, 0, 255, emptyfunc)
cv2.createTrackbar('upperGreen2', windowname2, 0, 255, emptyfunc)
cv2.createTrackbar('upperRed2', windowname2, 0, 255, emptyfunc)
cv2.createTrackbar('lowerBlue2', windowname2, 0, 255, emptyfunc)
cv2.createTrackbar('lowerGreen2', windowname2, 0, 255, emptyfunc)
cv2.createTrackbar('lowerRed2', windowname2, 0, 255, emptyfunc)
cap = cv2.VideoCapture(0)

   
while True:
    
    #################### Take each frame #####################3
    _,frame = cap.read()
    #resize input frame
    #frame = cv2.resize(frame, None, fx = 0.9, fy = 0.9,  interpolation = cv2.INTER_AREA)


    ##################### Convert BGR to HSV #####################33
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #cv2.imshow("abc", hsv)
    # Up_hue = cv2.getTrackbarPos('upperHue', windowname)
    # Up_saturation = cv2.getTrackbarPos('upperSaturation', windowname)
    # Up_value = cv2.getTrackbarPos('uppeValue', windowname)
    # lo_hue = cv2.getTrackbarPos('lowerHue', windowname)
    # lo_saturation = cv2.getTrackbarPos('lowerSaturation',windowname)
    # lo_value = cv2.getTrackbarPos('lowerValue', windowname)

    ################ FOR GREEN ####################
    up_b = cv2.getTrackbarPos('upperBlue', windowname)
    up_g = cv2.getTrackbarPos('upperGreen', windowname)
    up_r = cv2.getTrackbarPos('upperRed', windowname)
    lo_b = cv2.getTrackbarPos('lowerBlue', windowname)
    lo_g = cv2.getTrackbarPos('lowerGreen', windowname)
    lo_r = cv2.getTrackbarPos('lowerRed', windowname)

    ############# FOR RED ##############################
    upp_b = cv2.getTrackbarPos('upperBlue2', windowname2)
    upp_g = cv2.getTrackbarPos('upperGreen2', windowname2)
    upp_r = cv2.getTrackbarPos('upperRed2', windowname2)
    low_b = cv2.getTrackbarPos('lowerBlue2', windowname2)
    low_g = cv2.getTrackbarPos('lowerGreen2', windowname2)
    low_r = cv2.getTrackbarPos('lowerRed2', windowname2)



    ################### DEFINE RANGE OF COLOR IN HSV ####################################
    # upper_green = np.array([up_b, up_g, up_r])
    # lower_green = np.array([lo_r, lo_g, lo_r])
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    lower_green = np.array([45,100,20])
    upper_green = np.array([75,255,255])
    # lower_red = np.array([160,50,50])
    # upper_red = np.array([180,255,255])
    # lower_red = np.array([low_b, low_g, low_r])
    # upper_red = np.array([upp_b, upp_g, upp_r])
    # print(upper_green)
    # print(upper_red)

    ####################### THRESHOLD THE HSV IMAGE TO GET ONLY SELECTED COLORS #####################33
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
    mask2 = cv2.inRange(hsv, lower_green, upper_green)
    # mask3 = cv2.inRange(hsv, lower_red, upper_red)

    ################### Bitwise-AND mask and original image ########################
    res = cv2.bitwise_and(frame,frame, mask= mask)
    res = cv2.medianBlur(res, 5)
    res2 = cv2.bitwise_and(frame,frame, mask= mask2)
    res2 = cv2.medianBlur(res2, 5)

########################## CONTOURS OF GREEN AND RED OBJECTS ##################################
    #Read the image and convert it to grayscale
    gray1 = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)

    #Now convert the grayscale image to binary image
    ret, binary1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret, binary2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    contours1, hierarchy = cv2.findContours(binary1, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    contours2, hierarchy = cv2.findContours(binary2, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE) 

    # print("Length of contours {}".format(len(contours)))
    # print("Here is the hierarchy of detected contours :")
    # print(hierarchy)

    #res_copy = res.copy()

################### CENTER POINT OF GREEN COLOR OBJECT ########################################
    cx = 0
    cy = 0
    for contour in contours1:
        c = max(contours1, key = cv2.contourArea)
        (x,y,w,h) = cv2.boundingRect(c)
        M = cv2.moments(c)
        # print(M)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            print("p1:",cx, cy)
            cv2.circle(res, (cx, cy), 4, (0,0,255),-1)
            #print(f"x: {cx}, y: {cy}")
        #print("1",x,y,w,h)
        rect = cv2.rectangle(res, (x,y), (x+w,y+h), (255,0,0), 2)

        cv2.imshow('Drawn Contours1', rect)

#################### CENTER POINT OF GREEN COLOR OBJECT ########################################
    cx2 = 0
    cy2 = 0
    for contour in contours2:
        c2 = max(contours2, key = cv2.contourArea)
        (x2,y2,w2,h2) = cv2.boundingRect(c2)
        #print(x,y,w,h)

        M2 = cv2.moments(c2)
        if M2['m00'] != 0:
            cx2 = int(M2['m10']/M2['m00'])
            cy2 = int(M2['m01']/M2['m00'])
            print("p2:",cx2, cy2)
            cv2.circle(res2, (cx2, cy2), 4, (0,255,255),-1)
            #print(f"x: {cx}, y: {cy}")

        rect2 = cv2.rectangle(res2, (x2,y2), (x2+w2,y2+h2), (255,0,0), 2)
        #res_copy = cv2.drawContours(res_copy, contours, -1 , (255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow('Drawn Contours2', rect2)


####################### DRAW LINE BTWN TWO CENTER POINTS ###############################
    #img = np.zeros((512,512,3), np.uint8)
    line = cv2.line(frame, (cx,cy), (cx2,cy2), (0,0,255), 5)
    #cv2.imshow('line', line)

################### CENTER POINT OF A LINE ##########################3
    line1center = int((cx + cx2)/2)
    line2center = int((cy + cy2)/2)
    line_center = cv2.circle(frame, (line1center, line2center), 4, (255,255,0), -1)
    
    cv2.imshow('frame',line_center)
    #cv2.imshow('mask',mask)
    #cv2.imshow('mask3',mask3)
    #cv2.imshow('res2', res2)
    #cv2.imshow('res',res)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
