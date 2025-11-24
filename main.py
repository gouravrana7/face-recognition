import cv2
import os

cam = cv2.VideoCapture(0)
#cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(3,640) #width
cam.set(4,480)  #height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = input('enter name')
dataset_root = "dataset"
person_dir = os.path.join(dataset_root, face_id)
os.makedirs(person_dir, exist_ok=True)

count = 0
while(True):
    ret, img = cam.read()
    if not ret or img is None:
        print("Failed to capture frame; exiting.")
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray,1.3,5)

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1

        filename = f"{face_id}_{count}.jpg"
        cv2.imwrite(os.path.join(person_dir, filename), gray[y:y+h, x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >=50:
        break
print("exiting")
cam.release()
#cv2.cam.release()
cv2.destroyAllWindows()
