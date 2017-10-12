from __future__ import print_function
"""
This code will take input csv and convert it to a vector
for input in a net. Then the correctly shaped net will train
on input output pairs generated in batches, randomly
finally, the network is verified on some witheld data.

Make sure that the path for train_model point to whitespace delimited files that you want
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras import losses, layers

import pandas
import numpy as np


def train_model(filename):
	""" 
	traines a models on the given file, and returns it
	"""
	# import the csv as numpy array
	data = pandas.read_csv(filename, delim_whitespace=True, lineterminator='\n').values
	# shuffle the values in place, since data is clumped by class
	np.random.shuffle(data)
	
	# get all but last column for 'X', input data
	X = data[:, :-1] 
	X = X.astype(float)

	# get only the last row for 'y', targets and convert to 2-wide onehot
	y = data[:, -1]
	y = y.astype(int)	
	y_onehot = np.zeros((y.shape[0], 2))
	y_onehot[np.arange(y.shape[0]), y] = 1.
	y = y_onehot

	#create test data to check for overfitting
	X_test = X[:200]
	X = X[200:]	
	y_test = y[:200]
	y = y[200:]
	
	# num inputs is the number of columns in the csv excluding the last one
	num_inputs = X.shape[1]
	print('X shape: ', X.shape)
	print('y shape: ', y.shape)

	# create and compile the model
	model = Sequential()

	model.add(Dense(500, input_dim=num_inputs))
	model.add(Activation('relu'))
	model.add(Dropout(0.4))

	model.add(Dense(300))
	model.add(Activation('relu'))
	model.add(Dropout(0.4))

	model.add(Dense(10, name='learned_features'))
	model.add(Activation('relu'))

	model.add(Dense(2))
	model.add(Activation('softmax'))

	rms = RMSprop()
	loss = losses.categorical_crossentropy 
	model.compile(loss=loss, optimizer=rms, metrics=['accuracy'])

	# train the model
	model.fit(X, y, epochs=50, batch_size=100)

	# test the model for overfitting using withheld test data
	scores = model.evaluate(X_test, y_test)
	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

	return model

model1 = train_model('../rbp/cons_feature')
model1.save('model1.h5')
model2 = train_model('../rbp/rbp_feature')
model2.save('model2.h5')
print(model1.get_layer('learned_features'))
"""
Saved models:
name		training data	validation accuracy
model1.h5	cons_feature 	82.00%
model2.h5	rbp_feature		70.50%
"""
