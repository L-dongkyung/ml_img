from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np

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

# model train
model.fit(x_train, y_train, batch_size=32, epochs=50)

# model evaluate
score = model.evaluate(x_test, y_test)
print('loss=', score[0])
print('accuracy=', score[1])
