import cv2 as cv
import numpy as np
import time
import os
import HandTrackingModule as htm

color = (0, 255, 0)
brushThickness = 10
xp = 0
yp = 0
folderpath = "menuImg"
canvas = np.zeros((720, 1280, 3), np.uint8)
myList = os.listdir(folderpath)
print(myList)
overlayList = []
for impath in myList:
    image = cv.imread(f'{folderpath}/{impath}')
    overlayList.append(image)
menuImg = overlayList[0]

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 710)

detector = htm.handDetector(detectionCon=0.6)


while True:

    #import image
    success, img = cap.read()
    img = cv.flip(img,1)

    #Menu bar setting
    img[0:206, 75:1214] = menuImg
    #img = cv.addWeighted(img, 0.5, canvas, 0.5, 0)

    #cv.imshow("image", img)

    cv.waitKey(1)


    #Hand landmark identifying
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)

    if len(lmList) != 0 :
        #print(lmList)

        #tip of fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        #Checking upright fingers
        fingers = detector.fingerUp()
        print(fingers)

        #check mode
        if fingers[1] and fingers[2] == 1:

            xp, yp = x1, y1

            #print("Select mode")

            #Change display according to mode
            if y1<206:
                if 85<x1<400:
                    menuImg = overlayList[1]
                    color = (0, 0, 255)
                elif 450<x1<700:
                    menuImg = overlayList[0]
                    color = (0, 255, 0)
                elif 750<x1<900:
                    menuImg = overlayList[2]
                    color = (255, 0, 0)
                elif 950<x1<1150:
                    menuImg = overlayList[3]
                    color = (255, 255, 255)

            cv.circle(img, (x1, y1), 15, color, cv.FILLED)
            cv.circle(img, (x2, y2), 15, color, cv.FILLED)

        if fingers[1] and (not(fingers[0] or fingers[2] or fingers[3] or fingers[4])) == 1:

            cv.circle(img, (x1, y1), 10, color, cv.FILLED)
            if xp==0 and yp==0:
                xp,yp=x1,y1



            if color == (255,255,255):
                cv.line(img, (xp, yp), (x1, y1), (0,0,0), 50)
                cv.line(canvas, (xp, yp), (x1, y1), (0,0,0), 50)
            else:
                cv.line(img, (xp, yp), (x1, y1), color, brushThickness)
                cv.line(canvas, (xp, yp), (x1, y1), color, brushThickness)
                xp, yp = x1, y1

        xp, yp = x1, y1

    grayimg = cv.cvtColor(canvas, cv.COLOR_BGR2GRAY)
    _, InverseColorImg = cv.threshold(grayimg, 50, 265, cv.THRESH_BINARY_INV)
    InverseColorImg=cv.cvtColor(InverseColorImg, cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img,InverseColorImg)
    img = cv.bitwise_or(img,canvas)



    cv.imshow("image", img)
    #cv.imshow("canvas", canvas)