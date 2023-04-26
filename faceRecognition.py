import numpy as np
import cv2 as cv
import pickle

face_cascade = cv.CascadeClassifier("classifiers\\haarcascade_frontalface_alt2.xml")
eye_cascade = cv.CascadeClassifier("classifiers\\haarcascade_eye.xml")
smile_cascade = cv.CascadeClassifier("classifiers\\haarcascade_smile.xml")


recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("recognizers\\face-trainner.yml")

labels = {"person_name": 1}
with open("pickles\\face-labels.pickle","rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

capture = cv.VideoCapture(0)


while True:
    ret, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor= 1.5, minNeighbors = 5)
    for(x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w] #(ycord-start, xcord-end)
        roi_color = frame[y:y+h,x:x+w]

        id__, conf= recognizer.predict(roi_gray)
        if conf>=4 and conf <=85:
            font = cv.FONT_HERSHEY_SIMPLEX
            name = labels[id__]
            color = (255,255,255)
            stroke = 2
            cv.putText(frame, name, (x,y), font, 1, color, stroke, cv.LINE_AA)

        img_item = "my-image.png"
        cv.imwrite(img_item, roi_gray)

        color = (0,0,255) 
        stroke  = 2
        width = x+w #end_cord_x
        height = y+h #end_cord_y
        cv.rectangle(frame, (x,y), (width, height), color, stroke)
        
        
    cv.imshow("cam", frame)
    if cv.waitKey(20) & 0xFF== 27: #closes upon pressing the escape button
        break
capture.release()
cv.destroyAllWindows()