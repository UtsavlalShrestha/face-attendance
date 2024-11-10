import cv2
import numpy as np
import face_recognition


imgobama = face_recognition.load_image_file('images/obama.jpg')
imgobama = cv2.cvtColor(imgobama, cv2.COLOR_BGR2RGB)

imgtest = face_recognition.load_image_file('images/obamatestt.jpg')
imgtest = cv2.cvtColor(imgtest, cv2.COLOR_BGR2RGB)


faceloc = face_recognition.face_locations(imgobama)[0]
encodeobama = face_recognition.face_encodings(imgobama)[0]
cv2.rectangle(imgobama, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (255, 0, 255), 2)

faceloctest = face_recognition.face_locations(imgtest)[0]
encodetest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest, (faceloctest[3], faceloctest[0]), (faceloctest[1], faceloctest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeobama], encodetest)
facedis = face_recognition.face_distance([encodeobama], encodetest)
print(results, facedis)
cv2.putText(imgtest, f'{results} {round(facedis[0],2)}', (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0, 255),2)


cv2.imshow('Barack Obama', imgobama)
cv2.imshow('Test Image', imgtest)

cv2.waitKey(0)