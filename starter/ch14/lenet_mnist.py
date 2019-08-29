from imutils import paths
from keras.datasets import mnist
from keras.optimizers import SGD
import numpy as np
from sklearn.datasets import fetch_openml
from keras import backend as K
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from ch07.utilities.datasets import SimpleDatasetLoader
from ch07.utilities.preprocessing import SimplePreprocessor
from ch12.utilities.nn.conv.shallownet import ShallowNet
from ch12.utilities.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
import matplotlib.pyplot as plt

from ch14.lenet import LeNet

(X_train, y_train), (X_test, y_test) = mnist.load_data()
if K.image_data_format() == "channels_first":
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28, )
else:
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train/=255
X_test/=255


le = LabelBinarizer()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=128, epochs=20, verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(X_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in le.classes_]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
