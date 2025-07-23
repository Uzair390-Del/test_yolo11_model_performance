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

image=cv2.imread("resources\images\image1.jpg")
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
# results=model(image,save=True,conf=0.25,classes=[0],iou=0.3,save_txt=True, save_conf=True,save_crop=True)

#Object detection using yolo 
# results=model(image, conf=0.25, save=False)
# for result in results:
#     boxes=result.boxes
#     print(boxes)

# results=model(image, conf=0.25, save=False)
# for result in results:
#     boxes=result.boxes
#     for box in boxes:
#         x1,y1,x2,y2=box.xyxy[0]
#         print(f"X1: {x1}, y1: {y1},x2: {x2},y2:{y2}")

results=model(image, conf=0.25, save=False)
for result in results:
    boxes=result.boxes
    for box in boxes:
        x1,y1,x2,y2=box.xyxy[0]
        # converting the tensor into integer values 
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        print(f"X1: {x1}, y1: {y1},x2: {x2},y2:{y2}")
        # drawing rectangle arround detected images 
        cv2.rectangle(image,(x1,y1),(x2,y2),[256,0,0],1)


cv2.imshow("Image:",image)
cv2.waitKey(0)
# cv2.imshow()
