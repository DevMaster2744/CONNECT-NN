import numpy as np
import connectnn as cn
import time
from random import randrange
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib
import threading
import multiprocessing

matplotlib.use('TkAgg')
results_id = [0]
results = [0]

def neural_run():
    NN = cn.NeuralNetwork(1)
    NN.addLayer(170, cn.activationFunction.TANH)
    NN.addLayer(100, cn.activationFunction.TANH)
    NN.addLayer(17, cn.activationFunction.TANH)
    NN.addLayer(17, cn.activationFunction.TANH)
    NN.addLayer(5, cn.activationFunction.TANH)
    NN.addLayer(1, cn.activationFunction.SIGMOID)

    def decodeStr(_str: str):
        num = ""
        if str != "":
            for char in _str:
                num += str(ord(char))
                num += "0"
            return int(num)
        return 0

    frases_preconceituosas = [{"frase": "seu nego", "prec": True},
                            {"frase": "nego desgraçado", "prec": True},
                            {"frase": "nego vagabundo", "prec": True},
                            {"frase": "seu nego chato!", "prec": True},
                            {"frase": "seu negro burro", "prec": True},
                            {"frase": "vai se fuder negao", "prec": True},
                            {"frase": "a galinha josefa ta bem", "prec": False},
                            {"frase": "como vai o teu ego.", "prec": False},
                            {"frase": "gargalo eu ganho faca uai?", "prec": False},
                            {"frase": "Os nigocio vao bem na cidade ze", "prec": False},
                            {"frase": "oxi oxi", "prec": False},
                            {"frase": "seu viado", "prec": True},
                            {"frase": "ngo de merda", "prec": True},
                            {"frase": "eai, d boa!", "prec": False},
                            {"frase": "tu e gay", "prec": True},
                            {"frase": "vai morarr na favela", "prec": True},
                            {"frase": "vai tomar no cu seu bosta", "prec": True},
                            {"frase": "sabe o tio, ele ta meio mal la em casa...", "prec": False},
                            {"frase": "o joao ta trabalhando", "prec": False},
                            {"frase": "seu negro burro", "prec": True},
                            {"frase": "seu negro estupido", "prec": True},
                            {"frase": "lerda", "prec": True},
                            {"frase": "eu to de castigo", "prec": False},
                            ]

    for i in range(200):
        times = 0

        time.sleep(0.005)
        ran = randrange(1, len(frases_preconceituosas))
        varlouca = decodeStr(frases_preconceituosas[ran]["frase"])
        out = NN.run(np.array([((varlouca) / (math.ceil(varlouca / 10) * 10)),], dtype=np.float128))[0]

        out = round(out, 3)

        if randrange(1,5) == 1:
            pass#print(f"({out}, {decodeStr(frases_preconceituosas[ran]['frase'])}) - Result: {str(out > 0.6 and out >= 0.4)}, Correct: {str(frases_preconceituosas[ran]['prec'])}")

        if not (out <= 0.6 and out >= 0.4) and ((out > 0.6) == frases_preconceituosas[ran]["prec"]):
            global results_id
            global results

            results_id.append(i + 1)
            results.append(results[-1] + 1)
        else:
            IsNotReady = True

            while IsNotReady:
                time.sleep(0.005)
                varlouca = decodeStr(frases_preconceituosas[ran]["frase"])

                out = NN.run(np.array([((varlouca) / (math.ceil(varlouca / 10) * 10)),], dtype=np.float128))[0]

                out = round(out, 3)

                if randrange(1,5) == 1:
                    print(f"({out}, {decodeStr(frases_preconceituosas[ran]['frase'])}) - Result: {str(out > 0.5)}, Correct: {str(frases_preconceituosas[ran]['prec'])}")
                if not (out <= 0.6 and out >= 0.4) and ((out > 0.6) == frases_preconceituosas[ran]["prec"]):
                        IsNotReady = False
                elif out < 0.4 or out > 0.6:
                    NN.fit((-(0.601 - out) if frases_preconceituosas[ran]["prec"] else -(0.399 - out)), 0.01)
                else:
                    NN.fit((-(0.601) - out if frases_preconceituosas[ran]["prec"] else -(0.399 - out)), 0.1)
    while True:
        inp = input()
        if inp.lower() != "graph":
            varlouca = decodeStr(str(inp))
            out = NN.run(np.array([((varlouca) / (math.ceil(varlouca / 10) * 10)),], dtype=np.float128))[0]

            out = round(out, 3)

            time.sleep(0.5)
            print(out)

            print(str((out > 0.6) == frases_preconceituosas[ran]["prec"]))

def plot(x, y):
    plt.scatter(x, y, label="Results")
    plt.legend()
    plt.show()

# Define the graph function
def nn_run_mlp():
    if __name__ == "__main__":
            thread = threading.Thread(target=neural_run)
            thread.start()
            #thread.join()  # Ensure the process completes

nn_run_mlp()

while True:
    inp = input()
    if inp.lower() == "graph":
        print(f"{results_id}, {results}")
        plot(results_id, results)
    elif inp.lower() == "help":
        print("Type graph")
    time.sleep(0.5)