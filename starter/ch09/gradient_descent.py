import argparse
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def sigmoid_activation(x):
    return 1.0/(1+np.exp(-x))

def predict(X,W):
    preds = sigmoid_activation(X.dot(W))
    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1
    return preds

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100,help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01,help="learning rate")
args = vars(ap.parse_args())

(X,y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
print(y.shape)
y = y.reshape((y.shape[0],1))
print(y.shape)
print(X.shape)
X = np.c_[X, np.ones((X.shape[0]))]
print(X.shape)
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

print("[INFO] training...")
W = np.random.randn(X.shape[1], 1)
losses = []
for epoch in np.arange(0, args["epochs"]):
    preds = sigmoid_activation(trainX.dot(W))
    error = preds - trainY
    loss = np.sum(error ** 2)
    losses.append(loss)

    gradient = trainX.T.dot(error)
    W += -args["alpha"] * gradient

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1),loss))

print("[INFO] evaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))

plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY.reshape(500,), s=30)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()

