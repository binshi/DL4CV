from imutils import paths
from keras import Sequential
from keras import backend as K
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from ch07.utilities.datasets import SimpleDatasetLoader
from ch07.utilities.preprocessing import SimplePreprocessor
from ch12.utilities.nn.conv.shallownet import ShallowNet
from ch12.utilities.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        model=Sequential()
        inputShape=(height,width,depth)

        if K.image_data_format()=="channels_first":
            inputShape=(depth,height,width)

        model.add(Conv2D(20, (5, 5), padding="same",input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(20, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model


