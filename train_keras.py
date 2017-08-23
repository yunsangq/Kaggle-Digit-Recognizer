import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

seed = 777
np.random.seed(seed)

sample_fraction = 0.1
split_fraction = 0.1

epochs = 500

batch_size = 100

train = pd.read_csv('train.csv')

'''
if not sample_fraction > 0 and sample_fraction < 1:
    print('Data not sampled, all {} data points will be used!'.format(train.shape[0]))
else:
    print('Data sampled, {} data points will be used instead of all {} data points!'.format(round(train.shape[0] * sample_fraction), train.shape[0]))
    train = train.sample(frac=sample_fraction, random_state=seed)
'''
assert train.isnull().sum().sum() == 0, 'Null values found in train.'

y_train = train.iloc[:, 0].values.astype('uint8') # labels
X_train = train.iloc[:, 1:].values.astype('float32') # pixels

del train

X_train /= 255

X_train = X_train.reshape(-1, 28, 28, 1)

y_train = to_categorical(y_train, num_classes=10)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=split_fraction, random_state=seed)


model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='SAME', activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='SAME', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()

optimizer = RMSprop()
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=100, verbose=0, factor=0.5, min_lr=0.00001)

datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=False,
                             rotation_range=10,
                             zoom_range=0.1,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=False,
                             vertical_flip=False)

datagen.fit(X_train)

history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                              epochs=epochs,
                              validation_data=(X_valid, y_valid),
                              verbose=0,
                              steps_per_epoch=X_train.shape[0]//batch_size,
                              callbacks=[learning_rate_reduction])


acc = history.history['acc']
loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']
lr = history.history['lr']
for i in range(epochs):
    print('[%d] lr: %f train_loss: %f train_acc: %f, val_loss: %f val_acc: %f' %
          (i + 1, lr[i], loss[i], acc[i], val_loss[i], val_acc[i]))

scores = model.evaluate(X_train, y_train, verbose=0)
print('Baseline Error: {:.2f}%'.format(100-scores[1] * 100))

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')

