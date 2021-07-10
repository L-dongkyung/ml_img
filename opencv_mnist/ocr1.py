import sys
import numpy as np
import cv2

im = cv2.imread('numbers.png')
# 그레이스케일로 변환하고 블러르 걸고 이진화하기
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

#윤곽 추출하기
#cv4 버전을 사용할 경우 인덱스를 [0]으로 변경
#contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1] # 숫자 내부에 숫자도 검출
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1] # 숫자 내부검출x 가장 외곽만 검출

# 추출한 윤곽을 반복 처리하기
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if h< 20: continue
    red = (0, 0, 255)
    cv2.rectangle(im, (x, y), (x+w, y+h), red, 2)

cv2.imwrite('numbers-cnt.png', im)