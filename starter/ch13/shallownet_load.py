import cv2
from imutils import paths
from keras.models import load_model
from keras.optimizers import SGD
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from ch07.utilities.datasets import SimpleDatasetLoader
from ch07.utilities.preprocessing import SimplePreprocessor
from ch12.utilities.nn.conv.shallownet import ShallowNet
from ch12.utilities.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
import matplotlib.pyplot as plt

classLabels = ["cat", "dog", "panda"]
print("[INFO] sampling images...")
imagePaths = np.array(list(paths.list_images("../../../DL4CVStarterBundle/Datasets/animals")))
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths)
data = data.astype("float") / 255.0
print("[INFO] loading pre-trained network...")
model = load_model("shallownet_animals")

print("[INFO] predicting...")
preds = model.predict(data, batch_size=32).argmax(axis=1)

for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    cv2.putText(image, "Label: {}".format(classLabels[preds[i]]),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

