 #                      Created by Abhishek Bharadwaj
 #                      abhishekbharadwaj845@gmail.com
 #                      github @Abhishekbharadwaj106
 #                      Created using cv zone and Pipeline
 #       ____________Hand Gesture Detection Project with Cvzone updated package____________
# Go through the code and uncommentout the relevent things which is required in your relevent project,hope it will help

import cv2
from cvzone.HandTrackingModule import HandDetector
from scipy.stats import false_discovery_control

cap = cv2.VideoCapture(0)

detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    sucess, img = cap.read()
    hands,img = detector.findHands(img,flipType = True) #Drawing the outline
    # hands = detector.findHands(img,draw = False) #Not Drawing the outline
    # print(len(hands))

    if hands:
        #Hand1
        hand1 = hands[0]

        #importatn in many diffrent cases of developing a project

        lmList1 = hand1["lmList"] #list of 21 LANDMARK POINTS
        bbox1 = hand1["bbox"] # Bounding box (cordinates) x,y,w,h
        centerPoint1 = hand1["center"] # center of hand cx, cy
        handType1 = hand1["type"] #hand type left or right


        # print(len(lmList1),lmList1)
        # print(bbox1)
        # print(centerPoint1)
        # print(handType1)
        fingers1 = detector.fingersUp(hand1)

        if len(hands)==2:
            hand2 =hands[1]
            lmList2 = hand2["lmList"]  # list of 21 LANDMARK POINTS
            bbox2 = hand2["bbox"]  # Bounding box (cordinates) x,y,w,h
            centerPoint2 = hand2["center"]  # center of hand cx, cy
            handType2 = hand2["type"]  # hand type left or right

            # print(len(lmList2),lmList2)
            # print(bbox2)
            # print(centerPoint2)
            print(handType1,handType2)
            fingers2 = detector.fingersUp(hand2)
            print(fingers1,fingers2) #number of fingers up of both the hands
            # length, info, img = detector.findDistance(lmList1[8],lmList2[8],img) #Distance between two fingers with their points
            length, info, img = detector.findDistance(centerPoint1, centerPoint2, img) #Distance between two hands


    cv2.imshow("Image",img)
    cv2.waitKey(1)

    # Created By Abhishek Bharadwaj
    # Learning by Doing
