import os, sys
from keras.models import model_from_json
from imread import imread
from PIL import Image
import numpy as np
json_file = open("model40.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model40.h5")

print "Loaded model"

im = imread("/home/niraj/MachineLearning/MNIST/Images/rsz_3.jpg",opts={"strip_alpha":True})
im = im/255
predicted = loaded_model.predict(im.reshape((1, 28, 28, 1)))
print np.argmax(predicted)