import cv2
import numpy as np
import math

from picarx import Picarx

c=0
lk_params = {
    "winSize": (15, 15),  # 特徴点の計算に使う周辺領域サイズ
    "maxLevel": 2,  # ピラミッド数 (デフォルト0で、2の場合は1/4の画像まで使われる)
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # 探索アルゴリズムの終了条件
}

feature_params = {
    "maxCorners": 200,  # 特徴点の上限数
    "qualityLevel": 0.2,  # 閾値　（高いほど特徴点数は減る)
    "minDistance": 12,  # 特徴点間の距離 (近すぎる点は除外)
    "blockSize": 12  # 
}


color = np.random.randint(0, 255, (200, 3))
# VideoCapture オブジェクトを取得します
capture = cv2.VideoCapture(0)
ret, frame = capture.read()
ret, preframe = capture.read()
pregray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

px = Picarx()
mapimg = np.full((200,200,3),255,dtype=np.uint8)


px.set_camera_servo1_angle(0)
count=0

while(True):


    distance = px.ultrasonic.read()

    ret, frame = capture.read()

    mask = np.zeros_like(preframe)
    #mask = np.zeros_like(frame)


    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    p0 = cv2.goodFeaturesToTrack(pregray, mask=None, **feature_params)

    p1, status, err = cv2.calcOpticalFlowPyrLK(pregray, gray, p0, None, **lk_params)
    
    #sift = cv2.xfeatures2d.SIFT_create()
    #keypoints, descriptors = sift.detectAndCompute(frame, None)
    #img_sift = cv2.drawKeypoints(frame, keypoints, None, flags=4)
    #cv2.imshow('frame',img_sift)
    preframe = frame
    identical_p1 = p1[status==1]
    identical_p0 = p0[status==1]

    putpixel=0
    for i, (p1, p0) in enumerate(zip(identical_p1, identical_p0)):
        p1_x, p1_y = p1.astype(np.int).ravel()
        p0_x, p0_y = p0.astype(np.int).ravel()
        if p0_x > 260:
           if p0_x<480:
              mv = p0_x - p1_x
              mask = cv2.line(mask, (p1_x, p1_y), (p0_x, p0_y), color[1].tolist(), 2)
              putpixel=1
              #frame = cv2.circle(frame, (p1_x, p1_y), 5, color[1].tolist(), -1)

    if  putpixel==1:
      if  mv < 100:
         suitei = (mv) * distance / 496.386
         print( mv )
         c = c + math.atan( suitei / distance )
         print("c : ", c )
         count=count+1
         if count>100:
              count=0
              mapimg = np.full((200,200,3),255,dtype=np.uint8)
         x = int( distance * math.sin(c)+100)
         y = int(-distance * math.cos(c)+100)
         if x>0 and x<200 and y>0 and y<200:
             mapimg = cv2.rectangle(mapimg,(x,y),(x+1,y+1),color=(0, 255, 0))

    frame=cv2.add(frame,mask)
    cv2.imshow('frame',frame)
    cv2.imshow('map',mapimg)
    pregray = gray
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()