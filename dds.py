import cv2
import dlib
from scipy.spatial import distance
from playsound import playsound
import winsound
import streamlit as st


def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio

cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
face_cas=cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')
dlib_facelandmark = dlib.shape_predictor("tp\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat")
score=0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fac=face_cas.detectMultiScale(gray,1.1,5)
    for(x,y,w,h) in fac:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),4)
        print("Rectangle",x,y,w,h)
        roi=gray[y:y+h, x:x+w]

    faces = hog_face_detector(gray)
    print(faces)
    for face in faces:
        
        face_landmarks = dlib_facelandmark(gray,face)
        leftEye = []
        rightEye = []

        for n in range(1,67):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            cv2.circle(frame,(x,y),1,(255,0,0))

        for n in range(36,42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x,y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,255),1)

        for n in range(42,48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x,y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,255),1)

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)

        EAR = (left_ear+right_ear)/2
        EAR = round(EAR,2)
        
        if(EAR<0.21):
            score=score+1
            cv2.putText(frame,"Closed",(20,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
            cv2.putText(frame,'Score:'+str(score),(500,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            # if(rpred[0]==1 or lpred[0]==1):
        else:
            score=score-1
            cv2.putText(frame,"Open",(20,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
            if(score<1):
                score=0
                cv2.putText(frame,'Score:'+str(score),(500,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            else:
                cv2.putText(frame,'Score:'+str(score),(500,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        
        if(score>80):
            score=0
            winsound.PlaySound('TF016.WAV', winsound.SND_ASYNC)
        
        
    cv2.imshow("Are you Sleepy", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()