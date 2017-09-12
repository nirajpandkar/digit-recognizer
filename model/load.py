import numpy as np
import keras.models
# from keras.datasets import mnist
# from keras.utils import np_utils
from keras.models import model_from_json
from scipy.misc import imread, imresize,imshow
import tensorflow as tf

# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_test = X_test[5000:]
# y_test = y_test[5000:]

# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
# X_test = X_test / 255
# y_test = np_utils.to_categorical(y_test)
def init(): 
	json_file = open('model80.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	#load woeights into new model
	loaded_model.load_weights("model80.h5")
	print("Loaded Model from disk")

	#compile and evaluate loaded model
	#loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	#loss,accuracy = loaded_model.evaluate(X_test,y_test)
	#print('loss:', loss)
	#print('accuracy:', accuracy)
	graph = tf.get_default_graph()

	return loaded_model, graph