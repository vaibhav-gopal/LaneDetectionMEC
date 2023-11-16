import cv2 
import numpy as np

curveList = []
AvgNum = 10

def getLaneCurve(img, display = 0):
    imgCopy = img.copy()
    imgResult = img.copy()
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert images from BGR to HSV
    lowerWhite = np.array([0,0,150]) #lower values on HSV range
    upperWhite = np.array([179,30,255]) # higher values on HSV range
    maskWhite = cv2.inRange(imgHsv, lowerWhite, upperWhite) #makes video with mask given by previous range of values
    #cv2.imshow('thres', maskWhite)

    height, width, c = img.shape
    points = valTrackbars()
    originalWarp = warpImg(img, points, width, height)
    imgWarp = warpImg(maskWhite, points, width, height)

    imgWarpPoints = drawPoints(imgCopy,points)
    midPoint, imgHist = getHistogram(imgWarp, Display=True,minPers=0.5, region=4)
    basePoint, imgHist = getHistogram(imgWarp, Display=True,minPers=0.9)
    curveRaw = basePoint-midPoint
    
    curveList.append(curveRaw)
    if len(curveList)>AvgNum:
        curveList.pop(0)
    curve = int(sum(curveList)/len(curveList))
    
    if display != 0:
        imgInvWarp = warpImg(imgWarp, points, width, height, inv = True)
        imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
        imgInvWarp[0:height//3, 0:width] = 0,0,0
        imgLaneColor = np.zeros_like(img)
        imgLaneColor[:] =  0,255,0
        imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
        imgResult = cv2.addWeighted(imgResult,1,imgLaneColor,1,0)
        midY = 450
        cv2.putText(imgResult, str(curve), (width//2-80,85),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),3)
        cv2.line(imgResult, (width//2,midY),(width//2+(curve*3),midY), (255,0,255),5)
        cv2.line(imgResult, ((width//2 + (curve + 3)),midY-25),(width//2+(curve*3),midY+25), (0,255,0),5)
        for x in range(-30,30):
            w = width // 20
            cv2.line(imgResult, (w*x+int(curve//50), midY-10), (w*x+int(curve//50), midY+10), (0,0,255), 2)
        #fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        #cv2.PUTtEXT(imgResult, 'FPS '+ str(int(fps)), (20, 40),cv2.FONT_HERSHEY_SIMPLEX, 1, (230,50,50), 3);
    if display == 2:
        imgStacked = stackImages(0.7, ([img, imgWarpPoints, imgWarp], [imgHist, imgLaneColor, imgResult]))
        cv2.imshow("stacked result", imgStacked)
    elif display == 1:
        cv2.imshow("Result", imgResult)
    
    #cv2.imshow('warped', imgWarp)
    #cv2.imshow('points', imgWarpPoints)
    #cv2.imshow('histogram',imgHist)
    #cv2.imshow('warped test',originalWarp)
    
    curve = curve/100
    if curve>1: curve==1
    if curve< -1: curve ==-1
    
    return curve

def warpImg(img, points, width, height, inv = False): #function to transform image
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0], [width,0], [0,height], [width, height]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2,pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (width, height))
    return imgWarp

""" Due to the warpImg() funciton needing manual final warped coordinates I will also need to make 
    a function that allows me to change it in real time using trackbars"""

def nothing(x): #callback function of trackbar 
    return None
#creating window for trackbars
window = np.zeros((300, 512,3), np.uint8)
cv2.namedWindow('Trackbars')
#creating trackbar 
def initializeTrackbars(initialTrackbarVals,wT=480, hT=240):
    cv2.createTrackbar('Width Top', "Trackbars", initialTrackbarVals[0], wT//2, nothing)
    cv2.createTrackbar('Height Top', "Trackbars", initialTrackbarVals[1], hT, nothing)
    cv2.createTrackbar('Width Bottom', "Trackbars", initialTrackbarVals[2], wT//2, nothing)
    cv2.createTrackbar('Height Bottom', "Trackbars", initialTrackbarVals[3], hT, nothing)

def valTrackbars(wT=480, hT=240):
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([(widthTop, heightTop), (wT-widthTop, heightTop), 
                        (widthBottom, heightBottom), (wT-widthBottom, heightBottom)])
    return points

def drawPoints(img, points):
    for x in range(4):
        cv2.circle(img,(int(points[x][0]), int(points[x][1])), 15,(0,0,255), cv2.FILLED)
    return img

def getHistogram(img, minPers=0.1, Display=False, region=1):
    
    if region==1:
        histVals= np.sum(img, axis=0)
    else:
        histVals= np.sum(img[img.shape[0]//region:,:], axis=0)
        
    maxValue = np.max(histVals)
    minValue = minPers*maxValue #this is to remove noise from surrounding light and objects
    indexArray = np.where(histVals>=minValue)
    basePoint = int(np.average(indexArray))
    
    if Display:
        imgHist = np.zeros((img.shape[0], img.shape[1], 3),np.uint8)
        for x,intensity in enumerate(histVals):
            cv2.line(imgHist, (x,img.shape[0]), (x,int(img.shape[0]-intensity//255//region)), (255,0,255), 1)
            cv2.circle(imgHist,(basePoint, img.shape[0]),20,(0,255,255), cv2.FILLED)
        return basePoint, imgHist
    return basePoint

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

initialTrackbarValues = [16, 77, 0, 107]
initializeTrackbars(initialTrackbarValues)



cap = cv2.VideoCapture(0)

print("Curve values:\n")
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (480,240)) #frame sizes
    curve = getLaneCurve(frame, display=2)
    print("Curve is: ",curve, "\n")
    #cv2.imshow("video capture", frame)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()