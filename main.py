#FACE RECOGNITION + ATTENDANCE PROJECT
#REF - https://youtu.be/sz25xxF_AVE

import cv2
import numpy as np
import face_recognition

#load the images and convert into RGB

imgElon = face_recognition.load_image_file("images/elon1.jpg")
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file("images/bill_gates.jpg")
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

#detect face

faceloc = face_recognition.face_locations(imgElon)[0]
encodElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

facelocTest = face_recognition.face_locations(imgTest)[0]
encodTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,255),2)


#compare and find distance

results = face_recognition.compare_faces([encodElon],encodTest)
face_distance = face_recognition.face_distance([encodElon],encodTest)

print(results,face_distance)
cv2.putText(imgTest,f'{results}{round(face_distance[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


#print
cv2.imshow("Elon Musk", imgElon)
cv2.imshow("Elon Musk Test", imgTest)
cv2.waitKey(0)