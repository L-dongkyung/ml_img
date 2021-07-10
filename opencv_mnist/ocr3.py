import sys
import numpy as np
import cv2
import orc_mnist

# mnist 학습뎅터 읽어들이기
mnist = orc_mnist.build_model()
mnist.load_weights('mnist.hdf5')
# 이미지 읽어 들이기
im = cv2.imread('numbers100.png')

# 윤곽 추출
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
cv2.imwrite('number100-th.png')
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

# 추출한 좌표 정렬하기
rects = []
im_w = im.shape[1]
for i, cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cnt)
    if w< 10 or h<10: continue
    if w > im_w /5: continue
    y2 = round(y/ 10) *10 # y 좌포에 맞추기
    index = y2 * im_w + x
    rects.append((index, x, y, w, h))
rects = sorted(rects, key=lambda x:x[0])

# 해당 영역의 이미지 데이터 추출하기
X = []
for i, r in enumerate(rects):
    index, x, y, w, h = r
    num = gray[y:y+h, x:x+w]
    num = 255 - num
    # 정사각형 내부에 그림 옮기기
    ww = round((w if w>h else h) * 1.85)
    spc = np.zeros((ww,ww))
    wy = (ww - h)//2
    wx = (ww - w)//2
    spc[wy:wy+h, wx:wx+x] = num
    num = cv2.resize(spc, (28, 28))
    num = num.reshape(28*28)
    num = num.astype("float32") /255
    X.append(num)

# 예측하기
s = "31415926535897932384" + \
    "62643383279502884197" + \
    "16939937510582097494" + \
    "45923078164062862089" + \
    "98628034825342117067"
answer = list(s)
ok = 0
nlist = mnist.predict(np.array(X))
for i, n in enumerate(nlist):
    ans = n.argmax()
    if ans == index(answer[i]):
        ok +=1
    else:
        print("[ng", i, " 번째", ans, "!=", answer[i], np.int32(n*100))

print("정답률 : ", ok / len(nlist))