import keras
import tensorflow as tf
from keras.layers import Input, Dropout, Embedding, merge, LSTM, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential, load_model
import numpy as np
if __name__ == '__main__':
	x = np.load('case_b_input.npy')
	y = np.load('case_b_output_Fermi.npy')
	model = Sequential([
	  Embedding(input_dim=129, input_length=1024,
                      output_dim=64, name="embedding"),
	  LSTM(64, return_sequences=True, implementation=1, name="lstm_1"),
	  LSTM(64, implementation=1, name="lstm_2"),
	  BatchNormalization(),
	  Dense(32, activation="relu"),
	  Dense(6, activation="sigmoid")
	])
	model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

	model.fit(x, y, epochs=50, batch_size=1, verbose=1, shuffle=True)