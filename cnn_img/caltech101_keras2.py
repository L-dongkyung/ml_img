from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from PIL import Image
import os

np_load_old = np.load
np.load= lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

# category
categories = ["chair", "camera", "butterfly", "elephant", "flamingo"]
np_classes = len(categories)
# image size
image_w = 64
image_h = 64
# open data
x_train, x_test, y_train, y_test = np.load("./5obj.npy")
# 데이터 정규화
x_train = x_train.astype("float") / 256
x_test = x_test.astype("float") / 256
print('x_train shape:', x_train.shape)

# model 구축
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(np_classes))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# 추가
hdf5_file = "./image/5obj-model.hdf5"
if os.path.exists(hdf5_file):
    # 기존에 학습된 모델 읽어 들이기
    model.load_weights(hdf5_file)
else:
    # 학습한 모델을 파일로 저장하기
    model.fit(x_train, y_train, batch_size=32, epochs=50)
    model.save_weights(hdf5_file)

# model train
# model.fit(x_train, y_train, batch_size=32, epochs=50)

# model evaluate
score = model.evaluate(x_test, y_test)
print('loss=', score[0])
print('accuracy=', score[1])


# 추가
# predict
pre = model.predict(x_test)
# predict test
for i, v in enumerate(pre):
    pre_ans = v.argmax() # label pre
    ans = y_test[i].argmax() #label
    dat = x_test[i]  #img data
    if ans == pre_ans: continue
    # 예측이 틀리면 무엇이 틀렸는지 출력하기
    print("[NG]", categories[pre_ans], "!=", categories[ans])
    print(v)
    #이미지 출력하기
    fname = "image/error/" + str(i) + "-" + categories[pre_ans] + "-ne-" + categories[ans] + ".png"
    dat *= 256
    img = Image.fromarray(np.uint8(dat))
    img.save(fname)

