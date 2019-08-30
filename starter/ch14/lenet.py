from keras import Sequential
from keras import backend as K
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout


class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        model=Sequential()
        inputShape=(height,width,depth)

        if K.image_data_format()=="channels_first":
            inputShape=(depth,height,width)

        # model.add(Conv2D(20, (5, 5), padding="same",input_shape=inputShape))
        # model.add(Activation("relu"))
        # model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        # model.add(Conv2D(20, (5, 5), padding="same"))
        # model.add(Activation("relu"))
        # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # model.add(Flatten())
        # model.add(Dense(500))
        # model.add(Activation("relu"))
        # model.add(Dense(classes))
        # model.add(Activation("softmax"))
        # return model

        model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        # Fully connected layer
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(10))

        model.add(Activation('softmax'))

        return model