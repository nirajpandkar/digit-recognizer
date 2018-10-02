from flask import Flask, render_template, request, url_for

import re
import sys
import os
sys.path.append(os.path.abspath("./model"))

from load import *
model, graph = init()

def convertImage(imgData1):
	imgstr = re.search(r'base64,(.*)',imgData1).group(1)
	#print(imgstr)
	with open('output.png','wb') as output:
		output.write(imgstr.decode('base64'))


app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
	imgData = request.get_data()
	convertImage(imgData)
	x = imread('output.png',mode='L')
	x = np.invert(x)
	x = imresize(x,(28,28))
	#imshow(x)

	x = x.reshape(1,28,28,1)
	print "debug2"

	with graph.as_default():
		#perform the prediction
		out = model.predict(x)
		print(out)
		print(np.argmax(out,axis=1))
		print "debug3"
		#convert the response to a string
		response = np.array_str(np.argmax(out,axis=1))
		return response	


if __name__ == '__main__':
	app.debug = True
	app.run('0.0.0.0', port=5000)