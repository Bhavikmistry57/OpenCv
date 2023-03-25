# Python program to illustrate 
# arithmetic operation of
# addition of two images
    
# organizing imports 
import cv2 
import numpy as np 
from matplotlib import pyplot as plt

# path to input images are specified and  
# images are loaded with imread command 
    # image1 = cv2.imread('/home/milan/Original_Object_Images/1ec4ef0586b59e84.jpg') 
    # image2 = cv2.imread('/home/milan/Original_Object_Images/5b060a81384ceed3.jpg')
    
    # # cv2.addWeighted is applied over the
    # # image inputs with applied parameters
    # weightedSum = cv2.addWeighted(image1, 0.5, image2, 0.4, 0)
    
    # # the window showing output image
    # # with the weighted sum 
    # cv2.imshow('Weighted Image', weightedSum)
    
    # # De-allocate any associated memory usage  
    # if cv2.waitKey(0) & 0xff == 27: 
    #     cv2.destroyAllWindows() 

# subtraction of pixels of two images
  

# path to input images are specified and  
# images are loaded with imread command 
# image2 = cv2.imread('/home/milan/FastApi/segmentation/static/output/temp/0e8851ece9e32d54c1c21f012a9149fe83df5b74600f88f1.png') 
# image1 = cv2.imread('/home/milan/Original_Object_Images/0e8851ece9e32d54.jpg')
  
# # cv2.subtract is applied over the
# # image inputs with applied parameters
# sub = cv2.subtract(image1, image2)
  
# # the window showing output image
# # with the subtracted image 
# cv2.imshow('Subtracted Image', sub)
  
# # De-allocate any associated memory usage  
# if cv2.waitKey(0) & 0xff == 27: 
#     cv2.destroyAllWindows() 

#Accessing and Modifying pixel values

image1 = cv2.imread('/home/milan/Original_Object_Images/0e8851ece9e32d54.jpg')
# blue = image1[650,624,0]
# green = image1[650,624,1]
# red = image1[650,624,2]
# print(blue, green, red)
# image1[(20,100), (100,50)] = [255,255,255]
# print(image1.item(200,200,2))
# image1.itemset((200,200,2),100)
# print(image1.item(200,200,2))
# #IMAGE PROPERTIES:
# print("shape:", image1.shape, "size:", image1.size, "img-datatype:", image1.dtype )
# print(image1[(20,100), (100,50)])

# IMAGE ROI:
obj = image1[280:340, 330:390]
image1[273:333, 100:160] = obj

# #SPLITING AND MERGING IMAGE CHANNELS:
# #b,g = cv2.split(image1)
# image1[:,:,2] = 0
# #cv2.imshow('img',b)

# #image1 = cv2.merge((b,g))

cv2.imshow('image', image1)
cv2.waitKey(0)

#MAKING BORDERS FOR IMAGES(PADDING):

# BLUE = [255,0,0]
# image1 = cv2.imread('/home/milan/Original_Object_Images/0e8851ece9e32d54.jpg')

# replicate = cv2.copyMakeBorder(image1,10,10,10,10,cv2.BORDER_REPLICATE)
# reflect = cv2.copyMakeBorder(image1,10,10,10,10,cv2.BORDER_REFLECT)
# reflect101 = cv2.copyMakeBorder(image1,10,10,10,10,cv2.BORDER_REFLECT_101)
# wrap = cv2.copyMakeBorder(image1,10,10,10,10,cv2.BORDER_WRAP)
# constant= cv2.copyMakeBorder(image1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)

# cv2.imshow('img', constant)
# cv2.waitKey(0)

# plt.subplot(231),plt.imshow(image1,'gray'),plt.title('ORIGINAL')
# plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
# plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
# plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
# plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
# plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')

# plt.show()

#BITWISE OPRETIONS
# img1 = cv2.imread('/home/milan/Desktop/OpenCV/1bit1.png')  
# img2 = cv2.imread('/home/milan/Desktop/OpenCV/2bit2.png') 
  
# # cv2.bitwise_and is applied over the
# # image inputs with applied parameters 
# dest_and = cv2.bitwise_and(img2, img1, mask = None)
# dest_or = cv2.bitwise_or(img2, img1, mask = None)
# dest_xor = cv2.bitwise_xor(img1, img2, mask = None)
# dest_not1 = cv2.bitwise_not(img1, mask = None)
# dest_not2 = cv2.bitwise_not(img2, mask = None)
  
# # the window showing output image
# # with the Bitwise AND operation
# # on the input images
# cv2.imshow("a", img1)
# cv2.imshow("b", img2)
# cv2.imshow('Bitwise And', dest_and)
# cv2.imshow('Bitwise OR', dest_or)
# cv2.imshow('Bitwise XOR', dest_xor)
# cv2.imshow('Bitwise NOT on image 1', dest_not1)
# cv2.imshow('Bitwise NOT on image 2', dest_not2)

# if cv2.waitKey(0) & 0xff == 27: 
#     cv2.destroyAllWindows() 

#COLOR SPACES
# flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
# print (flags)
# image1 = cv2.imread('/home/milan/Original_Object_Images/0e8851ece9e32d54.jpg') 
# hsv = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
# gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# #cv2.imshow('img', hsv)
# cv2.imshow('gray', gray)
# cv2.waitKey(0)

#IMAGE THRESHOLDING:
#SIMPLE THRESHOLDING:

# img = cv2.imread('/home/milan/Desktop/OpenCV/cameraman.tif',0)
# ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
# ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
# ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
# ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

# #ADAPTIVE THRESHOLDING
# th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#             cv2.THRESH_BINARY_INV,11,2)
# th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,11,2)

# # Otsu's thresholding
# blur = cv2.GaussianBlur(img,(5,5),0)
# ret,O1 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# ret,O2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# ret,O3 = cv2.threshold(img,0,255,cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
# ret,O4 = cv2.threshold(img,0,255,cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
# ret,O5 = cv2.threshold(img,0,255,cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU)

# cv2.imshow('binary', thresh1)
# cv2.imshow('2', thresh2)
# cv2.imshow('3', thresh3)
# cv2.imshow('4', thresh4)
# cv2.imshow('5', thresh5)
# # cv2.imshow('A1', th2)
# # cv2.imshow('A2', th3) 
# # cv2.imshow('O1', O1)
# # cv2.imshow('O2', O2)
# # cv2.imshow('O3', O3)
# # cv2.imshow('O4', O4)
# # cv2.imshow('O5', O5)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#IMAGE BLURRING:
# image1 = cv2.imread('/home/milan/Desktop/OpenCV/taj.jpg')

# # 1) AVERAGING:
# blur = cv2.blur(image1,(5,5))
# cv2.imshow("avg", blur)

# #2) GAUSSIAN FILTERING:
# G_blur = cv2.GaussianBlur(image1,(5,5),0)
# cv2.imshow("Gaussian", G_blur)

# #3) MEDIAN FILTERING:
# M_blur = cv2.medianBlur(image1,5)
# cv2.imshow("median",M_blur)

# #4) BILATERAL FILTERING:
# B_blur = cv2.bilateralFilter(image1,9,75,75)
# cv2.imshow("Bilateral",B_blur)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

#MORPHOLOGICAL TRANSFORMATIONS:
# Morphological transformations are some simple operations based
# on the image shape. It is normally performed on binary images. 
# It needs two inputs, one is our original image, second one is called
# structuring element or kernel which decides the nature of operation.

# #1) EROSION:
# img = cv2.imread("/home/milan/Desktop/OpenCV/j.webp")
# kernel = np.ones((5,5),np.uint8)
# erosion = cv2.erode(img,kernel,iterations = 1)
# cv2.imshow("original", img)
# cv2.imshow("erosion", erosion)

# #2) DILATION:
# dilation = cv2.dilate(img,kernel,iterations = 1)
# cv2.imshow("dilation", dilation)

# #3) OPENING:
# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# cv2.imshow("opening", opening)

# #4) CLOSING:
# closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
# cv2.imshow("closing", closing)

# #5)MORPHOLOGICAL GRADIENT:
# gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
# cv2.imshow("gradient", gradient)

# #6)TOP HAT:
# tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
# cv2.imshow("tophat", tophat)

# #7)BLACK HAT:
# blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
# cv2.imshow("blackhat", blackhat)

# cv2.waitKey(0)

#EDGE DETECTION:

# image = cv2.imread('/home/milan/Original_Object_Images/0e8851ece9e32d54.jpg')
# image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # performing the edge detetcion
# #SOBEL FILTER:
# gradients_sobelx = cv2.Sobel(image, -1, 1, 0)
# gradients_sobely = cv2.Sobel(image, -1, 0, 1)
# gradients_sobelxy = cv2.addWeighted(gradients_sobelx, 0.5, gradients_sobely, 0.5, 0)

# #SCHARR FILTER:
# gradients_scharrx = cv2.Scharr(image, -1, 1, 0)
# gradients_scharry = cv2.Scharr(image, -1, 0, 1)
# gradients_scharrxy = cv2.addWeighted(gradients_scharrx, 0.5, gradients_scharry, 0.5, 0)

# #LAPLACIAN FILTER:
# gradients_laplacian = cv2.Laplacian(image, -1)

# #CANNY'S EDGE DETECTION:
# canny_output = cv2.Canny(image, 100, 180)

# cv2.imshow('Sobel x', gradients_sobelx)
# cv2.imshow('Sobel y', gradients_sobely)
# cv2.imshow('Sobel X+y', gradients_sobelxy)
# cv2.imshow('Scharr x', gradients_scharrx)
# cv2.imshow('Scharr y', gradients_scharry)
# cv2.imshow('Scharr X+y', gradients_scharrxy)
# cv2.imshow('laplacian', gradients_laplacian)
# cv2.imshow('Canny', canny_output)
# cv2.waitKey()

