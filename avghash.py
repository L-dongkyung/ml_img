from PIL import Image
import numpy as np

# 이미지를 이진수로 표현하여 형태확인 hash?사용

# 이미지 데이터를 average hash로 변환하기
def average_hash(fname, size=16):
    img = Image.open(fname)  # 이미지 데이터 열기
    img = img.convert('L')  # 그레이스케일로 변환하기
    img = img.resize((size, size), Image.ANTIALIAS)  # 리사이즈하기  Image.ANTIALIAS = img를 부드럽게 만드는 인자
    pixel_data = img.getdata()  # 픽셀 데이터 가져오기
    pixels = np.array(pixel_data)  # 넘파이 배열로 변환하기
    pixels = pixels.reshape((size, size))  # 2차원 배열로 변환하기
    avg = pixels.mean()  # 평균 구하기
    # print(pixels)  # img의 각 요소별 값 = 0~255 의 값
    # print(avg)  # 모든 요소의 평균??? - 차원 축소로 각 리스트의 평균값이 아닌 모든 데이터의 평균?
    # axis 0(column) or 1(row) 입력시 차원 축소 2차원 기준
    diff = 1* (pixels > avg)  # 평균보다 크면 1, 작으면 0으로 변환하기
    return diff

# 이진 해시로 변환하기
def np2hash(ahash):
    bhash = []
    for nl in ahash.tolist():
        s1 = [str(i) for i in nl]
        s2 = "".join(s1)
        i = int(s2,2)  # 이진수를 정수로 변환하기
        bhash.append("%04x" % i)
    return bhash

# Average Hash 출력하기
ahash = average_hash('tower.jpg', 24)
print(ahash)
print(np2hash(ahash))