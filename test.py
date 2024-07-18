from typing import Mapping
import connectnn as cn
import matplotlib
import matplotlib.pyplot as plt
from threading import Thread
import json
from random import randint
from time import sleep as wait
import numpy as np

matplotlib.use("TkAgg")

points_id = [0]
points = [0]

def setup_nn():
    print("Setup of ANN - Connect-NN")
    nn = cn.NeuralNetwork(1)

    nn.addLayer(170, cn.activationFunction.SIGMOID)
    nn.addLayer(50,  cn.activationFunction.SIGMOID)
    nn.addLayer(15, cn.activationFunction.SIGMOID)
    nn.addLayer(1, cn.activationFunction.SIGMOID)

    print("ANN - CONNECT-NN SETUP FINISHED")

    return nn

CONNECT_ANN = setup_nn()

def read_train_data():
    print("Reading TRAIN_DATA.JSON")
    return json.load(open("train_list.json"))

train_data = read_train_data()

def select_random_train_data(_train_data):
    phrases_len = len(_train_data["phrases"])

    rand_int = randint(0, phrases_len - 1)

    data = _train_data["phrases"][rand_int]
    return data["phrase"], data["isBad"], rand_int

def decode_str(str_: str):
    _str = ""
    for chr in str_:
        _str += str(ord(chr)) + "0"
    return np.float128(_str)

def CONNECT_ANN_Run():
    for i in range(1500):
        print(f"Time: {i + 1}")
        phrase, bad, id = select_random_train_data(train_data)

        decoded = decode_str(phrase)
        out = round(CONNECT_ANN.run([np.float128(decoded / ((len(str(decoded)) * 10) - 1))])[0], 3)
        if (out > 0.5) == bad:
            points_id.append(points_id[-1] + 1)
            points.append(points[-1] + 1)
        else:
            points_id.append(points_id[-1] + 1)
            points.append(points[-1] - 1)
            print("Started to train")
            while True:
                phrase, bad, id = select_random_train_data(train_data)

                decoded = decode_str(phrase)
                out = round(CONNECT_ANN.run([np.float128(decoded / ((len(str(decoded)) * 10) - 1))])[0], 3)
                if (out > 0.5) == bad:
                    print("Finished train" + f" out: {out}")
                    break
                else:
                    CONNECT_ANN.fit(-((0.501 if bad else 0.499) - out), 0.01)
                wait(0.01)

Thread(target=CONNECT_ANN_Run).start()

while True:
    inp = input()
    if inp.lower() == "graph":
        print("Show graph")
        plt.plot(points_id, points)
        plt.xlabel("Runs")
        plt.ylabel("Points")
        plt.show
        wait(1)