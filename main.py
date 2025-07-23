import cv2
from ultralytics import YOLO

# reading an image using OPENCV
# image=cv2.imread("resources\images\image1.jpg")
# model =YOLO("yolo11n.pt")
# detecting person 
# results=model(image,save=True,conf=0.25,classes=[0])

#detecting bus
# results=model(image,save=True,conf=0.25,classes=[5])

#detecting both person and  bus
# results=model(image,save=True,conf=0.25,classes=[0,5])

#adding maximum detection parametre
# results=model(image,save=True,conf=0.88,classes=[0],max_det=1)

image=cv2.imread("resources\images\image3.png")
model =YOLO("yolo11n.pt")

#adding NMS IOU
# decreasing iou wil remove overlaping bounding boxes
# results=model(image,save=True,conf=0.25,classes=[0],iou=0.9)

#add show image parametre 'show=true'
# results=model(image,save=True,conf=0.25,classes=[0],iou=0.3,show=True)

# add 'save_txt=True' , save detection results in a text file 

# results=model(image,save=True,conf=0.25,classes=[0],iou=0.3,save_txt=True)

# if you also want to save confidence value 
# results=model(image,save=True,conf=0.25,classes=[0],iou=0.3,save_txt=True, save_conf=True)

#if you want to save crop images of the detection 'save_crop=True'
results=model(image,save=True,conf=0.25,classes=[0],iou=0.3,save_txt=True, save_conf=True,save_crop=True)


# cv2.imshow("Image:",image)
# cv2.waitKey(0)
# cv2.imshow()
