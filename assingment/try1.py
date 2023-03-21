import os
import cv2
import numpy as np
# vid_cam = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

PATH = os.path.join("datasets", "images")
image_no = 0
for root, dirs, files in os.walk(PATH):
    # print(root, dirs, files)
    for file in files:
        if file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
                
            pil_image = cv2.imread(path) 
            gray = cv2.cvtColor(pil_image, cv2.COLOR_BGR2GRAY)
            size = (550, 550)
            image_array = np.array(gray, "uint8")
            faces = face_detector.detectMultiScale(image_array, 1.5, 5)
            for (x,y,w,h) in faces:
        
                cv2.rectangle(pil_image, (x,y), (x+w,y+h), (255,0,0), 2)
                cv2.imwrite(os.path.join(f"{image_no}.jpg"), pil_image)
                image_no += 1