import multiprocessing
import json
import multiprocessing
import wconnectnn as cn
import ctypes
import unidecode
import numpy as np
from time import sleep
from random import randint
from wcnnMainLib import select_random_train_data, train_data, ff_algorithm, bp_algorithm
def run_network(points, times, add_layers):
        #super(ConnectNetwork, self).__init__()

        times = times
        points = points
        
        #set_lrid = False
        #pointsId = points["LastRegisteredId"] + 1
        points["LastRegisteredId"] += 1
        results = []

        CONNECT_ANN = cn.NeuralNetwork(1)

        def buildNetwork():
            CONNECT_ANN.addLayer(170, cn.activationFunction.SIGMOID)
            CONNECT_ANN.addLayer(50,  cn.activationFunction.TANH)
            CONNECT_ANN.addLayer(15, cn.activationFunction.TANH)
            CONNECT_ANN.addLayer(1, cn.activationFunction.SIGMOID)

        if add_layers:
            buildNetwork()
        else:
            with open("best_nn.json") as f:
                data = json.load(f)
            CONNECT_ANN.makeFromDict(data)

        for i in range(times):
            print(f"RUN {i + 1}")
            phrase, bad, id = select_random_train_data(train_data)
            #points[pointsId].append(i)
            out = round(CONNECT_ANN.run(ff_algorithm(phrase))[0], 3)

            correct = (out > 0.5) == bad

            if not correct:
                CONNECT_ANN.fit(bp_algorithm(bad, out), 0.01)
            
            result = 1 if correct else -1

            results.append(results[-1] + result if len(results) > 0 else 0 + 1 if correct else result)
        pts_preset = points["results"]
        pts_preset.append({"ann_results": results, "cann": CONNECT_ANN})
        points.update({"results": pts_preset})
        return
        

if __name__ == "__main__":
    print("How many times per generation?")
    inp_times = int(input())

    print("How many generations?")
    gens = int(input())

    print("How many ANNS per generations?")
    ann_per_gen = int(input())

    multiprocessing.freeze_support()
    def generation(add_layers):
        with multiprocessing.Manager() as manager:
            points = manager.dict({"LastRegisteredId": -1, "results": []})
            processes = []

            for time in range(ann_per_gen):
                process = multiprocessing.Process(target=run_network, args=(points, inp_times, add_layers), name=f"CONNECT-NN {time}")
                process.start()
                processes.append(process)
                #sleep(0.25)
                

            for prcs in processes:
                prcs.join()
                print("################ENDED#####################")
            
            averages = []

            print(points)

            for results_dict in points["results"]:
                #results_dict = points["results"][i]
                averages.append({"average": np.average(results_dict["ann_results"]), "ann": results_dict["cann"]})

            averages.sort(key=lambda x: x["average"])
            print(f"Averages: {averages} // Best Average: {averages[-1]}")
            #averages[-1]["ann"].saveAsJson("best_nn.json")
        
    print("AddLayers? - TYPE Y OR N")

    alinp = input()

    if alinp.lower() == "y":
        generation(True)
    elif not alinp.lower() == "n":
        exit()
    
    for gen in range((gens - 1) if alinp.lower() == "y" else gens):
        generation(False)
            

