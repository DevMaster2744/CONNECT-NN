import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
import wcnnMainLib
import random
from time import sleep
from transformers import BertTokenizer
from tomllib import load as load_toml

# Sample data for text processing
phrases, Isbad = wcnnMainLib.getRspDataTables()
#Isbad = [1, 0, 1]  # Sample labels (1: bad, 0: good)

# Tokenizer setup

with open("config.toml", "rb") as fp:
    config = load_toml(fp)

class ReinforcedANN:
    def __init__(self, input_dim, output_dim, learning_rate=0.01):
        self.model = self.build_model(input_dim, output_dim, learning_rate)

    def build_model(self, input_dim, output_dim, learning_rate):
        if config["fromFile"]:
            model = load_model("connect.keras")
        else:
            model = Sequential()
            model.add(Input(shape=(input_dim,), name="input_layer"))
            model.add(Dense(64, activation='relu', name='dense_01'))
            model.add(Dense(32, activation='relu', name='dense_02'))
            model.add(Dense(output_dim, activation='sigmoid', name='sigmoid_dense'))
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, x_train, y_train, epochs=10):
        self.model.fit(x_train, y_train, epochs=epochs, verbose=1)

    def predict(self, state):
        pred = self.model.predict(state)
        return (pred > 0.5).astype(int)
    def save(self, sf: str):
        self.model.save(sf)

# Example usage
input_dim = 50  # Example input dimension should match tokenizer num_words

output_dim = 1  # Binary output
learning_rate = 0.001
ann = ReinforcedANN(input_dim, output_dim, learning_rate)

# Correct training loop

for _ in range(config["times"]):    
    rand_idx = random.randint(0, len(phrases) - 1)

    bad = Isbad[rand_idx]
    seq = wcnnMainLib.decode_str(phrases[rand_idx])

    for _ in range(50 - len(phrases[rand_idx].split(" "))):
        seq 

    x_train = seq.reshape(1, -1)  # Ensure correct shapes
    y_train = np.array([bad]).reshape(1, -1)
    ann.train(x_train, y_train, epochs=config["epochs"])

ann.save("connect.keras")

while True:
    sleep(0.1)
    print("\nTell me a word!")
    inp = input()
    print(ann.predict(wcnnMainLib.decode_str(inp).reshape(1, -1)))

# Predictions for sample data

#print(f"SEQUENCES: {sequences}")
