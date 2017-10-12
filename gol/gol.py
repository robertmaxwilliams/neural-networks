from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D
from keras.optimizers import RMSprop
from keras import losses, layers

def iterate(Z):
    # Count neighbours
    N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
         Z[1:-1,0:-2]                + Z[1:-1,2:] +
         Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])

    # Apply rules
    birth = (N==3) & (Z[1:-1,1:-1]==0)
    survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1)
    Z[...] = 0
    Z[1:-1,1:-1][birth | survive] = 1
    return Z


board = np.random.randint(2, size=(10, 10))



def generate_data():
	global board
	board = iterate(board)
	return board

def update(data):
    mat.set_data(data)
    return mat 

def data_gen():
    while True:
        yield generate_data()

fig, ax = plt.subplots()
mat = ax.matshow(generate_data())
def animate():
	plt.colorbar(mat)
	ani = animation.FuncAnimation(fig, update, data_gen, interval=50,
                              save_count=50)
	plt.show()

#now the fun part: take board_n as input, and board_n+1 as output.
# generate random board, zag back and forth 10 iterations,
# then make a new board
def generate_data(board_dims):
	X, y = list(), list()
	for n in range(10):
		board = np.random.randint(2, size=board_dims)
		for _ in range(10):
			X.append(board.copy())
			iterate(board)
			y.append(board.copy())
	X = np.array(X)
	y = np.array(y)
	X = np.expand_dims(X, 3)
	y = np.expand_dims(y, 3)
	print("dims:", X.shape, y.shape)
	return X, y
X, y = generate_data((20, 20))
model = Sequential()
model.add(Conv2D(20, (5,5), input_shape=(20,20, 1), strides=(1, 1), padding='same'))
#model.add(Activation('relu'))
model.add(Conv2D(20, (5,5), strides=(1, 1), padding='same'))
#model.add(Activation('relu'))
model.add(Conv2D(1, (5,5), strides=(1, 1), padding='same'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.fit(X, y, epochs=10, batch_size=10)
    
board = np.random.randint(2, size=(20,20))
board2 = board.copy()
board = np.expand_dims(board, 0)
board = np.expand_dims(board, 3)

for _ in range(10):
	print(np.reshape(board, (20, 20)))
	print(board2*2)
	print('\n')
	board = model.predict(board)
	board = np.rint(board)
	board = board.astype(int) 
	board2 = iterate(board2)
