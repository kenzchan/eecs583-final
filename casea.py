import keras
import tensorflow as tf
from keras.layers import Input, Dropout, Embedding, merge, LSTM, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential, load_model
import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from numpy.random import seed

if __name__ == '__main__':
	x = np.load('case_a_input.npy')
	y = np.load('case_a_output.npy')
	# kfold = KFold(n_splits=17, shuffle=True)
	# for i in range(5):
		# x, y = shuffle(x, y)
	
	seed(1)

	model = Sequential([
	  Embedding(input_dim=129, input_length=1024,
                      output_dim=64, name="embedding"),
	  LSTM(64, return_sequences=True, implementation=1, name="lstm_1"),
	  LSTM(64, implementation=1, name="lstm_2"),
	  BatchNormalization(),
	  Dense(32, activation="relu"),
	  Dense(1, activation="sigmoid")
	])
	opt = keras.optimizers.Adam(lr=0.001)
	model.compile(optimizer=opt, loss="mean_absolute_error", metrics=['accuracy'])

	model.fit(x, y, epochs=30, batch_size=32, verbose=1, shuffle=True,  validation_split = 0.2)
