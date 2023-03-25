import cv2
import numpy as np

img = np.zeros((512,512,3), np.uint8)



def func(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    # print(box1)
    # print(box2)

    image = cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 5)
    image = cv2.rectangle(img, (x3,y3), (x4,y4), (0,255,0), 5)
    cv2.imshow("img",image)

    x_inter1 = max(x1,x3)
    y_inter1 = max(y1,y3)

    x_inter2 = min(x2,x4)
    y_inter2 = min(y2,y4)

    ############## Calculating Area of Intersection ######################
    # width_inter = abs(x_inter2 - x_inter1)
    # height_inter = abs(y_inter2 - y_inter1)

    if x3 in range(x1, x2) or x4 in range(x1, x2):
    
        width_inter = abs(x_inter2 - x_inter1)
        height_inter = abs(y_inter2 - y_inter1)
    
    else:
        width_inter = 0
        height_inter = 0

    area_inter = width_inter * height_inter

    ################# Calculating Area of Union ########################
    width_box1 = abs(x2-x1)
    height_box1 = abs(y2-y1)

    width_box2 = abs(x4-x3)
    height_box2 = abs(y4-y3)

    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2

    area_union = area_box1 + area_box2 - area_inter

    ################# Calculate IOU ########################
    iou = area_inter/area_box1

    print(iou)

if __name__ == "__main__":
    #box1 = [156,156, 356,356]
    func(box1=[150,150, 250,50], box2=[150,100, 250,0])
    # func(box1=[150,150, 250,50], box2=[300,100, 400,50])
     

    

cv2.waitKey(0)

