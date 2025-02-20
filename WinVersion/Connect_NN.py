import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Input
from keras.optimizers import Adam
import wcnnMainLib
import random
from time import sleep
from tomllib import load as load_toml

# Train data (from wcnnMainLib) -> Win CONNECT-NN MAIN LIBRARY 
phrases, Isbad = wcnnMainLib.getRspDataTables()

print(f"Phrases: {phrases}")

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
        self.model.fit(x_train, y_train, epochs=epochs, verbose=0)

    def predict(self, state):
        pred = self.model.predict(state)
        return (pred > 0.5).astype(int)
    def save(self, sf: str):
        self.model.save(sf)

input_dim = 50

output_dim = 1
learning_rate = 0.001
ANN = ReinforcedANN(input_dim, output_dim, learning_rate)

def train():
    for _ in range(config["times"]):    
        rand_idx = np.random.randint(0, len(phrases) - 1)

        bad = Isbad[rand_idx]
        seq = wcnnMainLib.decode_str(phrases[rand_idx])
        print(phrases[rand_idx])
        print(bad)
        print(seq)
        #seq = wcnnMainLib.decode_str("legal")

        '''for _ in range(50 - len(phrases[rand_idx].split(" "))):
            seq'''

        x_train = seq.reshape(1, -1)  # Ensure correct shapes
        y_train = np.array([1 if bad else 0]).reshape(1, -1)
        ANN.train(x_train, y_train, epochs=config["epochs"])
        print(f"BAD: {bad}, ANN RESULT: {ANN.predict(x_train)}")

        if _ % 50 == 0:
            ANN.save("connect.keras")

    ANN.save("connect.keras")
def talk(phrase: str):
    pred = ANN.predict(wcnnMainLib.decode_str(phrase).reshape(1, -1))
    print(pred[0])
    return pred
'''
    while True:
        sleep(0.1)
        print("\nChat filter:")
        inp = input()
        print(ANN.predict(wcnnMainLib.decode_str(inp).reshape(1, -1)))

    # print(f"SEQUENCES: {sequences}")
'''