from __future__ import print_function
"""
Two neural nets are combined. Their input is now a concatination of
both input vectors, and their output is a concatination of their next-to-last layer
This is useful because the deep model contains some abstraction in the next-to-last
layer, otherwise the last softmax layer would not be able to accuratly classify the layer
"""
import keras
import numpy as np 

# load models created by make_models.py
model1 = keras.models.load_model('model1.h5')
model2 = keras.models.load_model('model2.h5')

#rename these, repeated names are not allowed 
# TODO fix make_models so it doesn't reuse names
layer_next_to_last1 = model1.get_layer(name='learned_features')
layer_next_to_last2 = model2.get_layer(name='learned_features')
layer_next_to_last1.name = 'learned_features1'
layer_next_to_last2.name = 'learned_features2'

#get the two layers that we want to use as inputs for the combiner
next_to_last1 = layer_next_to_last1.output
next_to_last2 = layer_next_to_last2.output

#get input layer tensor
input1 = model1.inputs[0]
input2 = model2.inputs[0]
print(model1.inputs)
print(next_to_last1)
from keras.models import Sequential, Model
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate

merge = concatenate([next_to_last1, next_to_last2])

model = Model(inputs=[input1, input2], outputs=merge)
model.compile(optimizer='adam', loss='binary_crossentropy',
               metrics=['accuracy'])

model.save('combined_model.h5')

# example showing that how inputs and output of the new net are formatted 
# pasted in example row just for simplicity of demonstration
data1 = str.split('3 0.487816329714 0.862462964392 0.508464285764 17.7556818182 14.9147727273 12.7840909091  11.3636363636')
data2 = str.split('0.00497159090909	0	0.00497159090909	0	0	0.00497159090909	0.00355113636364	0	0.0149147727273	0	0	0	0	0	0.00497159090909	0	0	0.0248579545455	0	0	0	0	0	0.00497159090909	0.0298295454545	0.00497159090909	0	0	0	0	0	0.0248579545455	0.00497159090909	0.00994318181818	0	0.00994318181818	0.00497159090909	0.0447443181818	0.00497159090909	0.00994318181818	0.0170454545455	0	0.0149147727273	0	0.00852272727273	0	0	0	0.00497159090909	0	0	0	0.0149147727273	0	0	0	0.00426136363636	0.00426136363636	0.00994318181818	0.0127840909091	0	0	0	0	0	0	0.00994318181818	0	0	0	0.00497159090909	0.00497159090909	0	0.00426136363636	0	0	0	0	0	0.00497159090909	0	0	0	0	0.00994318181818	0	0	0.00497159090909	0.00994318181818	0	0	0.0149147727273	0	0	0	0.00497159090909	0	0	0	0	0.00426136363636	0.0355113636364	0	0	0.0340909090909	0.0149147727273	0	0.0127840909091	0	0.00852272727273	0.00497159090909	0	0.00852272727273	0.00994318181818')

data1 = np.array([float(x) for x in data1])
data2 = np.array([float(x) for x in data2])
data1 = np.expand_dims(data1, 0)
data2 = np.expand_dims(data2, 0)
print('input 1: ', data1)
print('input 2: ', data2)
print('output: ', model.predict([data1, data2]))
# you can see the the output is 20 wide, a concatination of the two 10 wide
# layers in model1 and model2
