import cv2
import os
import albumentations as alb

def check_path(PATH):
    label = str(input("Enter filename: "))
    path = os.path.join(PATH, label)
    if os.path.exists(path) == False:
        os.makedirs(path)
    else: 
        return path
    return path
    

def get_data(cam_index=0):
    global PATH
    cap = cv2.VideoCapture(cam_index)
    image_no = 0
    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    while(True):
        ret, frame = cap.read()
        cv2.imshow('frame',frame)
    
        if cv2.waitKey(1) == 13: 
            cv2.imshow("captured", frame)
            faces = face_detector.detectMultiScale(frame, 1.3, 5)

            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
                image_no += 1
                cv2.imwrite(os.path.join(check_path(PATH), f"{image_no}.jpg"), frame[y:y+h,x:x+w])
        elif cv2.waitKey(1) == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
def Augmentation():
    augmentor = alb.Compose([ 
                            alb.RandomBrightnessContrast(p=0.2),
                            alb.GaussNoise(p=0.5),
                            alb.RandomGamma(p=0.2), 
                            alb.RGBShift(p=0.2), 
                            ],)
    for root, dirs, files in os.walk(PATH):
            # print(root, dirs, files)
        for file in files:
            if file.endswith("jpg"):
                path = os.path.join(root, file)
                image = cv2.imread(path)
                for i in range(10):
                    augmented = augmentor(image=image)
                    cv2.imwrite(os.path.join(root, f"{i}.jpg"), augmented["image"])
        # print(file)
PATH = os.path.join("datasets", "images")
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
get_data(0)
Augmentation()
    
