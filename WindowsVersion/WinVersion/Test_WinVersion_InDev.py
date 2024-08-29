import multiprocessing
import json
import multiprocessing
import wconnectnn as cn
#import ctypes
#import unidecode
import numpy as np
from time import sleep
from random import randint
from wcnnMainLib import select_random_train_data, train_data, ff_algorithm, bp_algorithm
import matplotlib.pyplot as plt

def run_network(points, lrid, times, add_layers, canPrint, linuxComp):
        #super(ConnectNetwork, self).__init__()

        times = times
        points = points
        
        #set_lrid = False
        #pointsId = points["LastRegisteredId"] + 1
        lrid.value += 1
        results = []

        CONNECT_ANN = cn.NeuralNetwork(2, linuxComp)

        def buildNetwork():
            CONNECT_ANN.addLayer(360, cn.activationFunction.SIGMOID)
            CONNECT_ANN.addLayer(75,  cn.activationFunction.TANH)
            CONNECT_ANN.addLayer(15, cn.activationFunction.TANH)
            CONNECT_ANN.addLayer(1, cn.activationFunction.SIGMOID)

        if add_layers:
            buildNetwork()
        else:
            with open("best_nn.json") as f:
                data = json.load(f)
            CONNECT_ANN.makeFromDict(data)

        for i in range(times):
            #print(f"RUN {i + 1}")
            phrase, bad, id = select_random_train_data(train_data)
            #points[pointsId].append(i)
            decoded_phrase = ff_algorithm(phrase)
            out = round(CONNECT_ANN.run([decoded_phrase, 0])[0], 3)

            correct = (out > 0.5) == bad

            if not correct:
                #CONNECT_ANN.fit(1, 0.1)
                CONNECT_ANN.fit(bp_algorithm(bad, out), 0.05)

            result = 1 if correct else 0

            results.append((result + results[-1]) if len(results) > 0 else 0)
            percentage = int((i / times) * 100)
            if canPrint:
                print(f"\r{percentage}% |{'█' * percentage + '-' * (100 - percentage)}| - OUT: {out}", end="\r")

        #pts_preset = points["results"]
        #pts_preset.append({"ann_results": results, "cann": CONNECT_ANN})
        #print(f"RESULTS: {results}")
        points.append({"ann_results": results, "cann": CONNECT_ANN})
        #return
        

if __name__ == "__main__":
    print("Use compatibility mode?")
    linuxComp_inp = input().lower()

    if linuxComp_inp == "y":
        linuxComp_inp = True
    elif linuxComp_inp == "n":
        linuxComp_inp = False
    else:
        exit()

    print("How many times per generation?") 
    inp_times = int(input())

    print("How many generations?")
    gens = int(input())

    print("How many ANNS per generations?")
    ann_per_gen = int(input())

    def generation(add_layers):
        with multiprocessing.Manager() as manager:
            multiprocessing.freeze_support()

            points = manager.list([])
            LastRegisteredId = manager.Value(typecode=int, value=-1)

            processes = []
            
            def start_process(canItPrint: bool):
                process = multiprocessing.Process(target=run_network, args=(points, LastRegisteredId, inp_times, add_layers, canItPrint, linuxComp_inp))
                process.start()
                processes.append(process)
                #sleep(0.25)
                return

            for time in range(ann_per_gen - 1):
                start_process(False)
                #sleep(0.25)

            start_process(True)
            
            for prcs in processes:
                prcs.join()

            #print(points)
            averages = []

            #print(points)

            for results_dict in points:
                #results_dict = points["results"][i]
                averages.append({"average": results_dict["ann_results"][-1], "ann": results_dict["cann"]})

            averages.sort(key=lambda x: x["average"])

            for i in range(ann_per_gen):
                print(f"// ANN {i} RESULT: {averages[i]['average']} //")
            
            #return averages[-1]["ann_results"]
            print(f"// FINAL RESULT: {averages[-1]['average']} ~= {int((averages[-1]['average'] / inp_times) * 100)}% //")
            averages[-1]["ann"].saveAsJson("best_nn.json")
        
    print("AddLayers? - TYPE Y OR N")

    alinp = input()
    #alinp = "n"

    if alinp.lower() == "y":
        generation(True)
    elif not alinp.lower() == "n":
        exit()
    
    for gen in range((gens - 2) if alinp.lower() == "y" else (gens - 1)):
        generation(False)
