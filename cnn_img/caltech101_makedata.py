from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split

# 분류 대상 카테고리 선택하기
caltech_dir = "./101_ObjectCategories"
categories = ["chair", "camera", "butterfly", "elephant", "flamingo"]
nb_classes = len(categories)

# img size
image_w = 64
image_h = 64
pixels = image_h * image_w * 3

X = []
Y = []
for idx, cat in enumerate(categories):
    # label
    lable = [0 for i in range(nb_classes)]
    lable[idx] = 1
    # img
    image_dir = caltech_dir + "/"+ cat
    files = glob.glob(image_dir+"/*.jpg")
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)
        X.append(data)
        Y.append(lable)
        if i%10 ==0:
            print(i, "\n", data)
X = np.array(X)
Y = np.array(Y)

# 학습 전용데이터와 테스트 데이터 구분
x_train, x_test, y_train, y_test = train_test_split(X, Y)
xy = (x_train, x_test, y_train, y_test)
np.save("./5obj.npy", xy)

print("ok,", len(Y))
