import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

#direct path to images folder
path = 'AttendanceImages'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

#use the names and import the imgs one by one

for cls in myList:
    currImage = cv2.imread(f'{path}/{cls}')
    images.append(currImage)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)

#encoding process

def findEncondings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #Find encodings
        encodeImgs = face_recognition.face_encodings(img)[0]
        encodeList.append(encodeImgs)

    return encodeList

#attendance process function

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

#call the function

encodeListKnown = findEncondings(images)
print('Encoding Complete')

#find matches between encodings

#1. initialize web cam

cap = cv2.VideoCapture(0)

#2. Get each frame
while True:
    success, img = cap.read()

#3. Reduce the size of img to help speed the process

    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

#4. Find location
    faceInCurrFrame = face_recognition.face_locations(imgS)
    encodesofCurrFrame = face_recognition.face_encodings(imgS,faceInCurrFrame)

#5. Find matches and compare to previous faces

    for encodeFace, faceLoc in zip(encodesofCurrFrame,faceInCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)

        #find distance
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
       # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)

            #create bounding interest
            #1. find loc

            y1,x2,y2,x1 = faceLoc

            # Reviving the image size
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

            #call attendance function whenever match is found

            markAttendance(name)

        cv2.imshow('webcam',img)
        cv2.waitKey(1)


# faceloc = face_recognition.face_locations(imgElon)[0]
# encodElon = face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)
#
# facelocTest = face_recognition.face_locations(imgTest)[0]
# encodTest = face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest,(facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,255),2)
#
#
# #compare and find distance
#
# results = face_recognition.compare_faces([encodElon],encodTest)
# face_distance = face_recognition.face_distance([encodElon],encodTest)