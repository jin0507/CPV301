import cv2
import os
import numpy as np
from PIL import Image
import pickle
import json

image_dir = os.path.join("datasets", "images")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for filename in os.listdir(os.path.join("datasets", "images")):
    for image in os.listdir(os.path.join("datasets", "images", filename)):
        label_path = os.path.join("datasets", "images", filename, f'{image.split(".")[0]}.json')
        coords = [0,0,0.00001,0.00001]
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                img_json = json.load(f)
                label = img_json["shapes"][0]["label"]
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]

            # id_ = label_ids[label]
                # print(label["shapes"][0]["label"])
            coords[0] = img_json['shapes'][0]['points'][0][0]
            coords[1] = img_json['shapes'][0]['points'][0][1]
            coords[2] = img_json['shapes'][0]['points'][1][0]
            coords[3] = img_json['shapes'][0]['points'][1][1]
            x, y, w, h = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            pil_image = Image.open(os.path.join("datasets", "images", filename, f'{image.split(".")[0]}.jpg'))
            size = (550, 550)
            # final_image = pil_image.resize(size, Image.LANCZOS)
            image_array = np.array(pil_image, "uint8")

            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

with open("face-labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("face-trainner.yml")

# print(label['shapes'][0]['points'][0][0])