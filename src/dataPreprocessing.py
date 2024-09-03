import cv2
import numpy as np
from glob import glob

jpgList = glob('../db/strawberry2/test/images/*.jpg')
print(jpgList)

for i in range(len(jpgList)):
    image_bgr = cv2.imread(jpgList[i]) # 이미지 파일 읽어들이기

    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV) # 이미지를 HSV으로 변환
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR) # 차원 변환

    # HSV로 색 추출 (0~15, 170~180)
    hsvLower1= np.array([0, 80, 80])    # 추출할 색의 하한(HSV)
    hsvUpper1 = np.array([15, 255, 255])    # 추출할 색의 상한(HSV)
    hsvLower2= np.array([170, 80, 80])    # 추출할 색의 하한(HSV)
    hsvUpper2 = np.array([180, 255, 255])    # 추출할 색의 상한(HSV)

    redMask1 = cv2.inRange(image_hsv, hsvLower1, hsvUpper1)    # HSV에서 마스크를 작성
    redMask2 = cv2.inRange(image_hsv, hsvLower2, hsvUpper2)    # HSV에서 마스크를 작성
    redMask = redMask1+redMask2
    mask_inv = cv2.bitwise_not(redMask)

    redFiltered = cv2.bitwise_or(image_bgr, image_bgr, mask=redMask) # 원래 이미지와 마스크를 합성
    grayFiltered = cv2.bitwise_or(image_gray, image_gray, mask=mask_inv)
    result = grayFiltered + redFiltered

    # cv2.imshow('test1_hsv', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    fileName = jpgList[i][30:]
    print(fileName)

    cv2.imwrite(f'../db/strawberry2/test/test/{fileName}', result)