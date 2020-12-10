import pandas as pd

GPU = ["Cypress", "Fermi", "Kepler", "Tahiti"]
for gpu in GPU:
    df = pd.read_csv("runtime_" + gpu + ".csv")
df["input"] = [ eval(i) for i in df["input"]]
print(df)
print(type(df["input"][0]))
val_dataframe = df.sample(frac=0.06)
train_dataframe = df.drop(val_dataframe.index)

# # Vocabulary has a padding character
# vocab_size = atomizer.vocab_size + 1

# # Language model. Takes as inputs source code sequences.
# seq_inputs = Input(shape=(1024,), dtype="int32")
# x = Embedding(input_dim=vocab_size, input_length=1024,
#                 output_dim=64, name="embedding")(seq_inputs)
# x = LSTM(64, return_sequences=True, implementation=1, name="lstm_1")(x)
# x = LSTM(64, implementation=1, name="lstm_2")(x)

# # Heuristic model. Takes as inputs the language model,
# #   outputs 1-of-6 thread coarsening factor
# x = BatchNormalization()(x)
# x = Dense(32, activation="relu")(x)
# outputs = Dense(6, activation="sigmoid")(x)

# self.model = Model(inputs=seq_inputs, outputs=outputs)
# self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
