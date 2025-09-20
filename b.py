# import cv2
# import numpy as np
# import face_recognition
# import os
# from datetime import datetime
# # from PIL import ImageGrab
 
# path = 'ImagesAttendance'
# images = []
# classNames = []
# myList = os.listdir(path)
# print(myList)
# for cl in myList:
#     curImg = cv2.imread(f'{path}/{cl}')
#     images.append(curImg)
#     classNames.append(os.path.splitext(cl)[0])
#     print(classNames)
 
# def findEncodings(images):
#     encodeList = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encode = face_recognition.face_encodings(img)[0]
#         encodeList.append(encode)
#     return encodeList
 
# def markAttendance(name):
#     with open('Attendance.csv','r+') as f:
#         myDataList = f.readlines()
#         nameList = []
#     for line in myDataList:
#         entry = line.split(',')
#         nameList.append(entry[0])
#     if name not in nameList:
#          now = datetime.now()
#          dtString = now.strftime('%H:%M:%S')
#          f.writelines(f'n{name},{dtString}')
 
# #### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# # def captureScreen(bbox=(300,300,690+300,530+300)):
# #     capScr = np.array(ImageGrab.grab(bbox))
# #     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
# #     return capScr
 
# encodeListKnown = findEncodings(images)
# print('Encoding Complete')
 
# cap = cv2.VideoCapture(0)
 
# while True:
#     success, img = cap.read()
# #img = captureScreen()
#     imgS = cv2.resize(img,(0,0),None,0.25,0.25)
#     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
 
#     facesCurFrame = face_recognition.face_locations(imgS)
#     encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
 
#     for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
#         matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
#         faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
# #print(faceDis)
#         matchIndex = np.argmin(faceDis)
 
#         if matches[matchIndex]:
#             name  = classNames[matchIndex].upper()
# #print(name)
#             y1,x2,y2,x1 = faceLoc
#             y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
#             cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
#             cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
#             cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
#             markAttendance(name)
 
# cv2.imshow('Webcam',img)
# cv2.waitKey(1)

import cv2

def compare_images(fake,real, dist_threshold=50, ratio_threshold=0.15):
    # Load images
    fake = cv2.imread("fk.png", cv2.IMREAD_GRAYSCALE)
    real = cv2.imread("rl.png", cv2.IMREAD_GRAYSCALE)

    # Use SIFT instead of ORB for better accuracy
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(fake, None)
    kp2, des2 = sift.detectAndCompute(real, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # Compute statistics
    if len(good) == 0:
        print("No matches found → FAKE")
        return

    avg_distance = sum([m.distance for m in good]) / len(good)
    match_ratio = len(good) / len(kp1)

    print(f"Average Good Match Distance: {avg_distance:.2f}")
    print(f"Good Match Ratio: {match_ratio:.2f}")

    # Decision logic (tuned thresholds)
    if avg_distance < dist_threshold and match_ratio > ratio_threshold:
        print("REAL (matches original closely)")
    else:
        print("FAKE (does not match original)")

    # Draw matches
    result = cv2.drawMatches(fake, kp1, real, kp2, good[:30], None, flags=2)
    cv2.imshow("Matches", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
compare_images("r.png", "f.png")
