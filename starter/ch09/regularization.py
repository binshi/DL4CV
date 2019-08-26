import argparse
import numpy as np
from imutils import paths
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from ch07.utilities.datasets import SimpleDatasetLoader
from ch07.utilities.preprocessing import SimplePreprocessor

# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
# args = vars(ap.parse_args())
# print("[INFO] loading images...")
# imagePaths = list(paths.list_images(args["dataset"]))
imagePaths = list(paths.list_images("../../../DL4CVStarterBundle/Datasets/animals"))

sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

le = LabelEncoder()
labels = le.fit_transform(labels)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=5)

for r in (None, "l1", "l2"):
    print("[INFO] training model with ‘{}‘ penalty".format(r))
    model = SGDClassifier(loss="log", penalty=r, max_iter=100, tol=1e-3, learning_rate="constant", eta0=0.01, random_state=42)
    model.fit(trainX, trainY)
    acc = model.score(testX, testY)
    print("[INFO] ‘{}‘ penalty accuracy: {:.2f}%".format(r, acc * 100))