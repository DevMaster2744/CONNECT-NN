import multiprocessing
import json
import multiprocessing
import wconnectnn as cn
import ctypes
import unidecode
import numpy as np
from random import randint
from wcnnMainLib import select_random_train_data, train_data, ff_algorithm, bp_algorithm
def run_network(points, times, add_layers):
        #super(ConnectNetwork, self).__init__()

        times = times
        points = points
        
        set_lrid = False
        points["LastRegisteredId"] += 1
        pointsId = points["LastRegisteredId"]
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
        pts_preset = dict(points["results"])
        pts_preset[pointsId] = {"id": pointsId, "ann_results": results}
        #points = pts_preset

        points.update({"LastRegisteredId": points["LastRegisteredId"],
                       "results": pts_preset})
        

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
            points = manager.dict({"LastRegisteredId": -1, "results": {}})
            processes = []

            points["LastRegisteredId"] = -1
            for time in range(ann_per_gen):
                process = multiprocessing.Process(target=run_network, args=(points, inp_times, add_layers), name=f"CONNECT-NN {time}")
                process.start()
                processes.append(process)

            for prcs in processes:
                prcs.join()
            
            averages = []

            print(points)

            for i in range(points["LastRegisteredId"] + 1):
                averages.append({"id": i, "average": np.average(points["results"][i]["ann_results"])})

            averages.sort(key=lambda x: x["average"])
            print(f"Averages: {averages} // Best Average: {averages[-1]}")
        
    print("AddLayers? - TYPE YES OR NO")

    alinp = input()

    if alinp.lower() == "yes":
        generation(True)
    elif not alinp.lower() == "no":
        exit()
    
    for gen in range(gens - 1):
        generation(False)
            

