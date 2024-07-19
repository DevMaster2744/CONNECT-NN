from typing import Mapping
import wconnectnn as cn
import matplotlib
import matplotlib.pyplot as plt
from threading import Thread
import json
from random import randint
from random import uniform
from time import sleep as wait
import numpy as np
import random
matplotlib.use("TkAgg")

points = []
results = []

nns = 0
points_graph_list = []

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
    return np.longdouble(_str)

def CONNECT_ANN_Run():
    global nns

    print("Setup of ANN - Connect-NN")
    nn = cn.NeuralNetwork(1)

    nn.addLayer(170, cn.activationFunction.SIGMOID)
    nn.addLayer(50,  cn.activationFunction.TANH)
    nn.addLayer(15, cn.activationFunction.TANH)
    nn.addLayer(1, cn.activationFunction.SIGMOID)

    print("ANN - CONNECT-NN SETUP FINISHED")

    nns += 1
    ann_nid = nns
    CONNECT_ANN = nn
    
    points.append({"id": ann_nid, "points_list": [0], "points_id": [0]})

    for i in range(1500):
        print(f"Time: {i + 1}")
        phrase, bad, id = select_random_train_data(train_data)

        decoded = decode_str(phrase)
        out = round(CONNECT_ANN.run([np.longdouble(decoded / ((len(str(decoded)) * 10) - 1))])[0], 3)
        if (out > 0.5) == bad:
            points[ann_nid - 1]["points_id"].append(points[ann_nid - 1]["points_id"][-1] + 1)
            points[ann_nid - 1]["points_list"].append(points[ann_nid - 1]["points_list"][-1] + 1)
            CONNECT_ANN.fit((0.501 if bad else 0.5) - out, 0.01)
        else:
            CONNECT_ANN.fit((0.501 if bad else 0.5) - out, 0.01)
            points[ann_nid - 1]["points_id"].append(points[ann_nid - 1]["points_id"][-1] + 1)
            points[ann_nid - 1]["points_list"].append(points[ann_nid - 1]["points_list"][-1] - 1)
            print("Started to train")
            while True:
                phrase, bad, id = select_random_train_data(train_data)

                decoded = decode_str(phrase)
                out = round(CONNECT_ANN.run([np.longdouble(decoded / ((len(str(decoded)) * 10) - 1))])[0], 3)
                if (out > 0.5) == bad:
                    print("Finished train" + f" out: {out}")
                    break
                elif out < 0.3:
                    CONNECT_ANN.fit((0.501 if bad else 0.5) - out, 0.1)
                elif out > 0.6:
                    CONNECT_ANN.fit((0.501 if bad else 0.5) - out, 0.1)
                else:
                    #CONNECT_ANN.fit((uniform(0.501, 0.539) - out if bad else out - uniform(0.460, 0.5)), 0.01)
                    CONNECT_ANN.fit((0.501 if bad else 0.5) - out, 0.01)
                wait(0.01)
    results.append(points[ann_nid - 1]["points_list"][-1])
    while True:
        inp = input()

        if int(inp.split("#")[0]) == ann_nid:
            decoded = decode_str(inp.split("#")[1])
            out = round(CONNECT_ANN.run([np.longdouble(decoded / ((len(str(decoded)) * 10) - 1))])[0], 3)

        print(f"Result: {str(out > 0.5)}")

def compare(x, y):
    if y > x:
        return True
    return False

Thread(target=CONNECT_ANN_Run).start()
Thread(target=CONNECT_ANN_Run).start()
Thread(target=CONNECT_ANN_Run).start()

while True:
    if len(results) >= 3:
        print("Show graph")
        for _result_list in points:
            plt.plot(_result_list["points_id"], _result_list["points_list"], label = f"ANN {_result_list['id']}")
        plt.xlabel("Runs")
        plt.ylabel("Points")
        plt.title("ANN Result List")
        plt.legend()
        plt.show()
        wait(0.5)
           


    wait(0.5)