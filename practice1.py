import cv2
import numpy as np
import matplotlib.pyplot as plt

#FOR IMAGE:
# img = cv2.imread("/home/milan/Original_Object_Images/4b519af5369c2b1d.jpg")
# B, G, R = cv2.split(img)
# print(img.shape)

# RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #plt.imshow(B)
# #plt.imshow(G)
# cv2.imshow("image", img)
# cv2.waitKey(0)&0xFF
# cv2.destroyAllWindows()
##################################################################################################################

#FOR VIDEO:
# cap = cv2.VideoCapture(0)

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('/home/milan/Desktop/OpenCV/output.avi',fourcc, 20.0, (640,480))

# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret==True:
#         #frame = cv2.flip(frame,0)

#         # write the flipped frame
#         out.write(frame)

#         cv2.imshow('frame',frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break

# # Release everything if job is finished
# cap.release()
# out.release()
# cv2.destroyAllWindows()
#################################################################################################################

# FOR DRAWING FUNCTION
    # img = np.zeros((512,512,3), np.uint8)
    # # Start coordinate, here (0, 0)
    # # represents the top left corner of image
    # start_point = (0, 0)
    
    # # End coordinate, here (250, 250)
    # # represents the bottom right corner of image
    # end_point = (155, 155)
    
    # # Green color in BGR
    # color = (170, 190, 100)
    
    # # Line thickness of 9 px
    # thickness = 5
    # #image = cv2.line(img, start_point, end_point, color, thickness)
    # image = cv2.line(img, (0,0), (511,511), (0,0,255), 5)
    # image = cv2.line(img, (0,511), (511,0), (0,0,255), 5)
    # image = cv2.rectangle(img, (156,156), (356,356), (255,0,0), -5)
    # image = cv2.circle(img,(256,256), 97, (0,255,0), -1)
    # image = cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
    # image = cv2.arrowedLine(image, start_point, end_point, color, thickness) 
    # pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
    # pts = pts.reshape((-1,1,2))
    # image2 = cv2.polylines(img,[pts],True,(0,255,255))
    # #image = cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA, True) 
    # image = cv2.putText(img, 'Opencv',(167,383), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 2, cv2.LINE_AA, False)
    # image = cv2.putText(img, 'Opencv',(167,383), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 2, cv2.LINE_AA, True)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)&0xFF
    # cv2.destroyAllWindows()

# Draw a triangle with centroid using OpenCV

    # width = 400
    # height = 300
    
    # # Create a black window of 400 x 300
    # img = np.zeros((height, width, 3), np.uint8)
    
    # # Three vertices(tuples) of the triangle 
    # p1 = (100, 200)
    # p2 = (50, 50)
    # p3 = (300, 100)
    
    # # Drawing the triangle with the help of lines
    # #  on the black window With given points 
    # # cv2.line is the inbuilt function in opencv library
    # cv2.line(img, p1, p2, (255, 0, 0), 3)
    # cv2.line(img, p2, p3, (255, 0, 0), 3)
    # cv2.line(img, p1, p3, (255, 0, 0), 3)
    
    # # finding centroid using the following formula
    # # (X, Y) = (x1 + x2 + x3//3, y1 + y2 + y3//3) 
    # centroid = ((p1[0]+p2[0]+p3[0])//3, (p1[1]+p2[1]+p3[1])//3)
    
    # # Drawing the centroid on the window  
    # cv2.circle(img, centroid, 4, (0, 255, 0))
    
    # # image is the title of the window
    # cv2.imshow("image", img)
    # cv2.waitKey(0)

#################################################################################################################################

#Mouse as a Paint-Brush
# events = [i for i in dir(cv2) if 'EVENT' in i]
# print (events)
# mouse callback function
    # drawing = False # true if mouse is pressed
    # mode = True # if True, draw rectangle. Press 'm' to toggle to curve
    # ix,iy = -1,-1

    # def draw_circle(event,x,y,flags,param):
    #     global ix,iy,drawing,mode

    #     if event == cv2.EVENT_LBUTTONDBLCLK:
    #         cv2.circle(img,(x,y),100,(255,0,0),-1)
        
    #     elif event == cv2.EVENT_LBUTTONDOWN:
    #         drawing = True
    #         ix,iy = x,y

    #     elif event == cv2.EVENT_MOUSEMOVE:
    #         if drawing == True:
    #             if mode == True:
    #                 cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
    #             else:
    #                 cv2.circle(img,(x,y),50,(0,0,255),-1)

    #     elif event == cv2.EVENT_LBUTTONUP:
    #         drawing = False
    #         if mode == True:
    #             cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
    #         else:
    #             cv2.circle(img,(x,y),5,(0,0,255),-1)

    # # Create a black image, a window and bind the function to window
    # img = np.zeros((512,512,3), np.uint8)
    # cv2.namedWindow('image')
    # cv2.setMouseCallback('image',draw_circle)


    # while(1):
    #     cv2.imshow('image',img)
    #     k = cv2.waitKey(1) & 0xFF
    #     if k == ord('m'):
    #         mode = not mode
    #     elif k == 27:
    #         break

    # cv2.destroyAllWindows()
##########################################################################################################################

# OpenCV BGR color palette with trackbars

# empty function called when
# any trackbar moves
def emptyFunction():
    pass
   
def main():
      
    # blackwindow having 3 color chanels
    image = np.zeros((512, 512, 3), np.uint8) 
    windowName ="Open CV Color Palette"
      
    # window name
    cv2.namedWindow(windowName) 
       
    # there trackbars which have the name
    # of trackbars min and max value 
    cv2.createTrackbar('upperHue', windowName, 153,180, emptyFunction)
    cv2.createTrackbar('upperSaturation', windowName, 255,255, emptyFunction)
    cv2.createTrackbar('upperValue', windowName, 255,255, emptyFunction)
    # cv2.createTrackbar('lowerHue', windowName, 64,180, emptyFunction)
    # cv2.createTrackbar('lowerSaturation', windowName, 72,255, emptyFunction)
    # cv2.createTrackbar('lowerValue', windowName, 49,255, emptyFunction)
    # cv2.createTrackbar('Blue', windowName, 0, 255, emptyFunction)
    # cv2.createTrackbar('Green', windowName, 0, 255, emptyFunction)
    # cv2.createTrackbar('Red', windowName, 0, 255, emptyFunction)
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, windowName, 0,1,emptyFunction)
    # Used to open the window
    # till press the ESC key
    while(True):
        cv2.imshow(windowName, image)
          
        if cv2.waitKey(1) == 27:
            break
          
        # values of blue, green, red
        Up_hue = cv2.getTrackbarPos('upperHue', windowName)
        Up_saturation = cv2.getTrackbarPos('upperSaturation', windowName)
        Up_value = cv2.getTrackbarPos('uppeValue', windowName)
        lo_hue = cv2.getTrackbarPos('lowerHue', windowName)
        lo_saturation = cv2.getTrackbarPos('lowerSaturation',windowName)
        lo_value = cv2.getTrackbarPos('lowerValue', windowName)
        b = cv2.getTrackbarPos('Blue', windowName)
        g = cv2.getTrackbarPos('Green', windowName)
        r = cv2.getTrackbarPos('Red', windowName)
          
        # merge all three color chanels and
        # make the image composites image from rgb   
        # image[:] = [blue, green, red]
        # print(blue, green, red)
        if switch == 0:
            image[:] = 0
        else:
            image[:] = [b,g,r,Up_hue, Up_saturation, Up_value,lo_hue, lo_saturation, lo_value]

           
    cv2.destroyAllWindows()
  
# Calling main()         
if __name__=="__main__":
    main()