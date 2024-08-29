import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
 
video_capture = cv2.VideoCapture(0)
 
shivam_image = face_recognition.load_image_file("imgFaceRecog/shivam.jpg")
shivam_encoding = face_recognition.face_encodings(shivam_image)[0]

#aditya_image = face_recognition.load_image_file("imgFaceRecog/adittya.jpg")
#aditya_encoding = face_recognition.face_encodings(aditya_image)[0]
 
animesh_image = face_recognition.load_image_file("imgFaceRecog/animesh.jpg")
animesh_encoding = face_recognition.face_encodings(animesh_image)[0]
 
chandan_image = face_recognition.load_image_file("imgFaceRecog/chandan.jpg")
chandan_encoding = face_recognition.face_encodings(chandan_image)[0]

digvijay_image = face_recognition.load_image_file("imgFaceRecog/digvijay.jpg")
digvijay_encoding = face_recognition.face_encodings(digvijay_image)[0]

parmanik_image = face_recognition.load_image_file("imgFaceRecog/parmanik.jpg")
parmanik_encoding = face_recognition.face_encodings(parmanik_image)[0]

#jayant_image = face_recognition.load_image_file("imgFaceRecog/jayant.jpg")
#jayant_encoding = face_recognition.face_encodings(jayant_image)[0]

Narendra_image = face_recognition.load_image_file("imgFaceRecog/Narendra.jpg")
Narendra_encoding = face_recognition.face_encodings(Narendra_image)[0]

#ravisir_image = face_recognition.load_image_file("imgFaceRecog/ravisir.jpg")
#ravisir_encoding = face_recognition.face_encodings(ravisir_image)[0]

 
known_face_encoding = [
shivam_encoding,
#aditya_encoding,
animesh_encoding,
chandan_encoding,
digvijay_encoding,
parmanik_encoding,
#jayant_encoding,
Narendra_encoding,
#ravisir_encoding
]
 
known_faces_names = [
"shivam",
#"aditya",
"animesh",
"chandan",
"digvijay",
"parmanik",
#"jayant",
"Narendra",
#"ravisir"
]
 
students = known_faces_names.copy()
 
face_locations = []
face_encodings = []
face_names = []
s=True
 
now = datetime.now()
current_date = now.strftime("%d-%b-%y ")

f = open(current_date+'.csv','w+',newline ='')
lnwriter = csv.writer(f)
 
while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]
 
            face_names.append(name)
            if name in known_faces_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,100)
                fontScale              = 1.5
                fontColor              = (255,0,0)
                thickness              = 3
                lineType               = 2
 
                cv2.putText(frame,name+' Present', 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
 
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H:%M:%S")
                    lnwriter.writerow([name,current_time])
    cv2.imshow("attendence system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
video_capture.release()
cv2.destroyAllWindows()
f.close()


