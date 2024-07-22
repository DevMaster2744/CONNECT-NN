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
import unidecode
import ctypes

matplotlib.use("TkAgg")

points = []
results = []

nns = 0
finished_nns = 0
points_graph_list = []

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

def prevent_shutdown():
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

def allow_shutdown():
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)

def read_train_data():
    print("Reading TRAIN_DATA.JSON")
    return json.load(open("train_list.json"))

train_data = read_train_data()

def select_random_train_data(_train_data):
    phrases_len = len(_train_data["phrases"])

    rand_int = randint(0, phrases_len - 1)

    data = _train_data["phrases"][rand_int]
    return unidecode.unidecode(data["phrase"]), data["isBad"], rand_int

def decode_str(str_: str):
    _str = ""
    for wrd in str_:
        for chr in wrd:
            _str += str(ord(chr)) + "0"
        _str += str(ord(chr)) + "00"
    return np.longdouble(_str)

print("How many times the ANN must train?")
tinp = int(input())

class CONNECT_ANN():
    def __init__(self, **kwargs) -> None:
        self.CONNECT_ANN = cn.NeuralNetwork(1)

        if not 'addLayers' in kwargs:
            self.CONNECT_ANN.addLayer(170, cn.activationFunction.SIGMOID)
            self.CONNECT_ANN.addLayer(50,  cn.activationFunction.TANH)
            self.CONNECT_ANN.addLayer(15, cn.activationFunction.TANH)
            self.CONNECT_ANN.addLayer(1, cn.activationFunction.SIGMOID)
        elif kwargs["addLayers"] == True:
            self.CONNECT_ANN.addLayer(170, cn.activationFunction.SIGMOID)
            self.CONNECT_ANN.addLayer(50,  cn.activationFunction.TANH)
            self.CONNECT_ANN.addLayer(15, cn.activationFunction.TANH)
            self.CONNECT_ANN.addLayer(1, cn.activationFunction.SIGMOID)

    def run(self):
        global points
        global finished_nns
        global nns
        global tinp

        print("Setup of ANN - Connect-NN")
        print("ANN - CONNECT-NN SETUP FINISHED")

        nns += 1
        ann_nid = nns
        
        points.append({"id": ann_nid, "points_list": [0], "points_id": [0]})

        for i in range(tinp):
            print(f"Time: {i + 1}")
            phrase, bad, id = select_random_train_data(train_data)

            decoded = decode_str(phrase)
            out = round(self.CONNECT_ANN.run([np.longdouble(decoded / ((len(str(decoded)) * 10) - 1))])[0], 3)
            if (out > 0.5) == bad:
                points[ann_nid - 1]["points_id"].append(points[ann_nid - 1]["points_id"][-1] + 1)
                points[ann_nid - 1]["points_list"].append(points[ann_nid - 1]["points_list"][-1] + 1)
                self.CONNECT_ANN.fit((0.501 if bad else 0.5) - out, 0.01)
            else:
                self.CONNECT_ANN.fit((0.501 if bad else 0.5) - out, 0.01)
                points[ann_nid - 1]["points_id"].append(points[ann_nid - 1]["points_id"][-1] + 1)
                points[ann_nid - 1]["points_list"].append(points[ann_nid - 1]["points_list"][-1] - 1)
                print("Started to train")
                while True:
                    phrase, bad, id = select_random_train_data(train_data)

                    decoded = decode_str(phrase)
                    out = round(self.CONNECT_ANN.run([np.longdouble(decoded / ((len(str(decoded)) * 10) - 1))])[0], 3)
                    if (out > 0.5) == bad:
                        print("Finished train" + f" out: {out}")
                        break
                    elif out < 0.3:
                        self.CONNECT_ANN.fit((0.501 if bad else 0.5) - out, 0.1)
                    elif out > 0.6:
                        self.CONNECT_ANN.fit((0.501 if bad else 0.5) - out, 0.1)
                    else:
                        #CONNECT_ANN.fit((uniform(0.501, 0.539) - out if bad else out - uniform(0.460, 0.5)), 0.01)
                        self.CONNECT_ANN.fit((0.501 if bad else 0.5) - out, 0.01)
                    wait(0.01)
        finished_nns += 1
        return
    
    def talk(self, _str):
        decoded = decode_str(_str)
        out = round(self.CONNECT_ANN.run([np.longdouble(decoded / ((len(str(decoded)) * 10) - 1))])[0], 3)
        print("ALERT!" if (out > 0.5) else "OK")
        return out > 0.5
    
def compare(x):
    return x["points_list"][-1]

print("How many Threads you want? - DANGER! - THIS VALUE CAN CRASH THE PROGRAM")
maxn = int(input())

def train(fromGenetic: bool):
    global finished_nns
    global nns
    #global results
    global points

    nns = 0
    anns = []
    #results.clear()
    points.clear()
    finished_nns = 0

    for _ in range(maxn) :
        if fromGenetic:
            anns.append({"ann": CONNECT_ANN(addLayers = False), "id": _ + 1})
            data = {}
            with open("bestNN.json", 'r') as f:
                data = json.load(f)
                print(f"Layers: {len(data['layers'])}")
            anns[-1]["ann"].CONNECT_ANN.makeFromDict(data)
        else:
            anns.append({"ann": CONNECT_ANN(), "id": _ + 1})
        Thread(target=anns[-1]["ann"].run).start()

    while finished_nns < maxn:
        wait(0.1)

    print(f"Best network: {points[-1]['id']}")
    print(f"Points: {points[-1]['points_list']}")

    print("Talk with the ANN")

    points = sorted(points, key=compare)

    '''
    for _ in range(6):
        inps = input()
        anns[points[-1]['id']]["ann"].talk(inps)
        print(f"{5 - _} remain")
    '''

    anns[points[-1]['id'] - 1]["ann"].CONNECT_ANN.saveAsJson("bestNN.json")

allow_shutdown()

if __name__ == "__main__":
    try:
        prevent_shutdown()

        train(fromGenetic=False)

        for i in range(4):
            train(fromGenetic=True)

        for _result_list in points:
            plt.plot(_result_list["points_id"], _result_list["points_list"], label = f"ANN {_result_list['id']}")
    finally:
        allow_shutdown()

        plt.xlabel("Runs")
        plt.ylabel("Points")
        plt.title("ANN Result List")
        plt.legend()
        plt.show()
        plt.clf()