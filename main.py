import cv2
from ultralytics import YOLO

# reading an image using OPENCV
image=cv2.imread("resources\images\image1.jpg")
model =YOLO("yolo11n.pt")
results=model(image,save=True,conf=0.80)
# cv2.imshow("Image:",image)
# cv2.waitKey(0)
cv2.imshow()
