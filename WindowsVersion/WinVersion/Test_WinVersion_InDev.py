import numpy as np
from keras.models import Sequential
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

class ReinforcedANN:
    def __init__(self, input_dim, output_dim, learning_rate=0.01):
        self.model = self.build_model(input_dim, output_dim, learning_rate)

    def build_model(self, input_dim, output_dim, learning_rate):
        model = Sequential()
        model.add(Input(shape=(input_dim,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(output_dim, activation='sigmoid'))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, x_train, y_train, epochs=10):
        self.model.fit(x_train, y_train, epochs=epochs, verbose=1)

    def predict(self, state):
        pred = self.model.predict(state)
        return (pred > 0.5).astype(int)

# Example usage
input_dim = 50  # Example input dimension should match tokenizer num_words

output_dim = 1  # Binary output
learning_rate = 0.001
ann = ReinforcedANN(input_dim, output_dim, learning_rate)

# Correct training loop

with open("config.toml", "rb") as fp:
    config = load_toml(fp)

for _ in range(config["times"]):    
    rand_idx = random.randint(0, len(phrases) - 1)

    bad = Isbad[rand_idx]
    seq = wcnnMainLib.decode_str(phrases[rand_idx])

    for _ in range(50 - len(phrases[rand_idx].split(" "))):
        seq 

    x_train = seq.reshape(1, -1)  # Ensure correct shapes
    y_train = np.array([bad]).reshape(1, -1)
    ann.train(x_train, y_train, epochs=config["epochs"])

while True:
    sleep(0.1)
    print("\nTell me a word!")
    inp = input()
    print(ann.predict(wcnnMainLib.decode_str(inp).reshape(1, -1)))

# Predictions for sample data

#print(f"SEQUENCES: {sequences}")
