from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# from utilties.preprocessing import SimplePreprocessor
# from datasets import SimpleDatasetLoader
from imutils import paths
import argparse

from ch07.utilities.datasets import SimpleDatasetLoader
from ch07.utilities.preprocessing import SimplePreprocessor

# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
# ap.add_argument("-k", "--neighbors", required=True, help="# of nearest neighbors for classification")
# ap.add_argument("-j", "--jobs", required=True, help="# of jobs for k-NN distance (-1 uses all available cores)")
# args = vars(ap.parse_args())
# print("Loading images")
# imagePaths = list(paths.list_images(args["dataset"]))

imagePaths = list(paths.list_images("../../../DL4CVStarterBundle/Datasets/animals"))

sp = SimplePreprocessor(32,32)
sdl = SimpleDatasetLoader(preprocessors=[sp])

(data,labels) = sdl.load(imagePaths,verbose=500)
print(data.shape)
data = data.reshape((data.shape[0], 3072))
print(data.shape)

print("features matrix: {:.1f}MB".format(data.nbytes/(1024*1000.0)))

le = LabelEncoder()
labels = le.fit_transform(labels)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

print("[INFO] evaluating k-NN classifier...")
# model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
model = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX),target_names=le.classes_))
