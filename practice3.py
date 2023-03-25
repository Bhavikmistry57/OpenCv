import cv2
import numpy as np

#Read the image and convert it to grayscale
# image = cv2.imread('/home/milan/Desktop/OpenCV/contour.jpg')
# image = cv2.resize(image, None, fx=0.6,fy=0.6)

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# #Now convert the grayscale image to binary image
# ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# #Now detect the contours
# #RETR_EXTERNAL: and CHAIN_APPROX_SIMPLE:
# #We can say, under this law, Only the eldest in every family is taken care of. It doesn’t care about other members of the family :).
# #contours, hierarchy = cv2.findContours(binary, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE) 

# #RETR_LIST: and CHAIN_APPROX_TC89_L1:
# # Parents and kids are equal under this rule, and they are just contours.
# #contours, hierarchy = cv2.findContours(binary, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_TC89_L1) 

# #RETR_TREE:
# # It even tells, who is the grandpa, father, son, grandson and even beyond... :).
# contours, hierarchy = cv2.findContours(binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_TC89_KCOS) 
# # contours, hierarchy = cv2.findContours(binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE) 

# #Visualize the data structure
# print("Length of contours {}".format(len(contours)))
# print("Here is the hierarchy of detected contours :")
# print(hierarchy)

# # draw contours on the original image
# image_copy = image.copy()
# image_copy = cv2.drawContours(image_copy, contours, -1 , (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

# #Visualizing the results
# cv2.imshow('Grayscale Image', gray)
# cv2.imshow('Drawn Contours', image_copy)
# cv2.imshow('Binary Image', binary)

# cv2.waitKey(0)
# cv2.destroyAllWindows()



blank = np.zeros((720,720,3), np.uint8)
cv2.rectangle(blank,(168,95),(2,20),(0,0,255),3)
cv2.rectangle(blank,(366,345),(40,522),(0,255,0),3)

rect1x, rect1y = ((168+2)/2, (95+20)/2)
rect2x, rect2y = ((366+40)/2, (345+522)/2)
rect1center = int(rect1x),int(rect1y)
rect2center = int(rect2x),int(rect2y)
print(rect1center)
print(rect2center)
cv2.line(blank, (rect1center), (rect2center), (0,0,255), 4)

line1center = int((rect1x+rect2x)/2)
print(line1center)
line2center = int((rect1y+rect2y)/2)
print(line2center)

# center = (line1center/2,line2center/2)
# print(center)
cv2.circle(blank, (line1center, line2center), 4, (0,255,255),-1)

cv2.imshow('test', blank)
cv2.waitKey(0)
cv2.destroyAllWindows()