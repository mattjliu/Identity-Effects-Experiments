import numpy as np
import pandas as pd
import os
import argparse

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CV model Prediction Options')
    parser.add_argument('-w', dest='weights_file', required=True,
                        help='Weights file')
    parser.add_argument('-d', dest='dataset', choices=['TEST', 'TRAIN'], required=True,
                        help='Predict on MNIST test or MNIST train set')
    parser.add_argument('-f', dest='out_file', required=True,
                        help='Output filpath for CV predictions')

    args = parser.parse_args()

    num_classes = 10

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    if args.dataset == 'TEST':
        x, y = x_test, y_test
    else:
        x, y = x_train, y_train

    model.load_weights(args.weights_file)
    softmaxes = model.predict(x)
    predictions = np.argmax(softmaxes, axis=1)
    labels = np.argmax(y, axis=1)
    data = np.concatenate((softmaxes,
                           predictions.reshape((x.shape[0], 1)),
                           labels.reshape((x.shape[0], 1))), axis=1)
    output = pd.DataFrame(data, columns=[str(i) for i in range(10)] + ['pred', 'label'])

    try:
        os.makedirs(os.path.dirname(args.out_file))
    except FileExistsError:
        pass
    output.to_csv(args.out_file)
    print('Done')
