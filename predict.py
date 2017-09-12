import os, sys
from keras.models import model_from_json
from keras.datasets import mnist
from imread import imread
from PIL import Image
import numpy as np
json_file = open("model80.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model80.h5")

print("Loaded model")

# im = imread("./Images/rsz_2.jpg",opts={"strip_alpha":True})
# im = im/255
# predicted = loaded_model.predict(im.reshape((1, 28, 28, 1)))
# print(np.argmax(predicted))


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_test = X_test[5000:]
y_test = y_test[5000:]

predicted = loaded_model.predict()
predictions = [np.argmax(x) for x in predicted]
observed = [np.argmax(y) for y in y_test]
accuracy = numpy.mean(predictions == observed)
print("Test Accuracy: %.2f%%" % (accuracy*100))
