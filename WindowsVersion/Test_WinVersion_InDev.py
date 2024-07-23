import multiprocessing
import json
import multiprocessing
import wconnectnn as cn
import ctypes
import unidecode
import numpy as np
from random import randint

def bp_algorithm(cond: bool, out: float):
    return (0.501 if cond else 0.5) - out

def ff_algorithm(phrase: str):
    decoded = decode_str(phrase)
    return [np.longdouble(decoded / ((len(str(decoded)) * 10) - 1))]

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

def prevent_shutdown():
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

def allow_shutdown():
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)

def read_train_data():
    with open("train_list.json") as f:
        data = json.load(f)
    return data

train_data = read_train_data()

def select_random_train_data(_train_data):
    phrases_len = len(_train_data["phrases"])

    rand_int = randint(0, phrases_len - 1)

    data = _train_data["phrases"][rand_int]
    return unidecode.unidecode(data["phrase"]), data["isBad"], rand_int

def decode_str(str_: str):
    out = np.longdouble(0)
    for wrd in str_:
        for chr in wrd:
            chr_ord = ord(chr)
            out = (out * (10 ** len(str(chr_ord))) + ord(chr))
    return out

class ConnectNetwork(multiprocessing.Process):
    def __init__(self, times: int, points, **kwargs):
        super(ConnectNetwork, self).__init__()

        self.times = times
        self.points = points

        self.points.append([])
        self.pointsId = len(self.points) - 1

        self.CONNECT_ANN = cn.NeuralNetwork(1)

        def buildNetwork():
            self.CONNECT_ANN.addLayer(170, cn.activationFunction.SIGMOID)
            self.CONNECT_ANN.addLayer(50,  cn.activationFunction.TANH)
            self.CONNECT_ANN.addLayer(15, cn.activationFunction.TANH)
            self.CONNECT_ANN.addLayer(1, cn.activationFunction.SIGMOID)

        if not 'addLayers' in kwargs:
            buildNetwork()
        elif kwargs["addLayers"] == True:
            buildNetwork()
        else:
            with open("best_nn.json") as f:
                data = json.load(f)
            self.CONNECT_ANN.makeFromDict(data)

    def run(self):
        for i in range(self.times):
            print(f"RUN {i + 1}")
            phrase, bad, id = select_random_train_data(train_data)
            #self.points[self.pointsId].append(i)
            out = round(self.CONNECT_ANN.run(ff_algorithm(phrase))[0], 3)
            if (out > 0.5) != bad:
                self.CONNECT_ANN.fit(bp_algorithm(bad, out), 0.01)
            print(f"OUT: {out}")
    def saveJson(self):
        self.CONNECT_ANN.saveAsJson("best_nn.json")

print("How many times per generation?")
inp_times = int(input())

print("How many generations?")
gens = int(input())

print("How many ANNS per generations?")
ann_per_gen = int(input())

if __name__ == "__main__":
    multiprocessing.freeze_support()

    with multiprocessing.Manager() as manager:
        points = manager.list()

        processes = []

        for gen in range(gens):
            for time in range(ann_per_gen):
                process = ConnectNetwork(inp_times, points, addLayers=False)
                process.start()
                processes.append(process)
            for task in processes:
                task.join()
            
            averages = []

            for ann_points in points:
                averages.append()
            

