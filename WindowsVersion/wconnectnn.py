import numpy as np
from random import uniform
from enum import Enum
import json
#from numpyencoder import NumpyEncoder

class activationFunction(Enum):
    SIGMOID = 1
    BINARY = 2
    TANH = 3

def activationFunctionFromEnum(x: np.longdouble, activationFunctionType: activationFunction) -> float:
    if activationFunctionType == activationFunction.SIGMOID:
        return np.longdouble(1 / (1 + np.exp(-x, dtype=np.longdouble)))
    
    if activationFunctionType == activationFunction.TANH:
        th = (np.longdouble(np.exp(x, dtype=np.longdouble) - np.exp(-x, dtype=np.longdouble)) / (np.exp(x, dtype=np.longdouble) + np.exp(-x, dtype=np.longdouble)))
        return th
    
    if activationFunctionType == activationFunction.BINARY:
        return 1 if (1 / (1 + np.exp(-x, dtype=np.longdouble)) > 0.5) else 0

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.longdouble):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)

class unit():
    def __init__(self, inputSize: int, activationFunctionType: activationFunction, **kwargs) -> None:
        if not 'fromDict' in kwargs:
            self.weights = [uniform(-1,1) for _ in range(inputSize)]
        else:
            #print(np.longdouble(kwargs["fromDict"]["weight"]))
            self.weights = ([np.longdouble(wh) for wh in kwargs["fromDict"]["weight"]])
        self.activationFunctionType = activationFunctionType
        self.inputs = []
        self.output = 0
    def run(self, inputs: tuple) -> float:
        self.inputs = inputs

        result = np.dot(inputs, self.weights)

        result = activationFunctionFromEnum(result, self.activationFunctionType)
        return result

    def fit(self, error: float, alpha: float):
        for i in range(len(self.weights)):
            self.weights[i] +=  activationFunctionFromEnum(self.inputs[i], self.activationFunctionType) * error * alpha
    '''def setInputSize(self, inputSize: int):
        self.inp'''


class layer():
    def __init__(self, inputSize: int, layerSize: int, activationFunctionType: activationFunction, **kwargs) -> None:
        if not 'fromDict' in kwargs:
            self.units = [unit(inputSize=inputSize, activationFunctionType=activationFunctionType) for _ in range(layerSize)]
            self.activationFunctionType = activationFunctionType
        else:
            self.units = [unit(inputSize=inputSize, activationFunctionType=activationFunction(activationFunctionType), fromDict=kwargs['fromDict']['neurons'][_]) for _ in range(layerSize)]
            self.activationFunctionType = activationFunction(activationFunctionType)
    def fit(self, error: float, alpha: float):
        for unit in self.units:
            unit.fit(error, alpha)
    def run(self, inputs: tuple):
        result = []
        for unit in self.units:
            result.append(unit.run(inputs))
        return result
    def _get_units_len_(self):
        return len(self.units)

class NeuralNetwork():
    def __init__(self, initialInputSize: int) -> None:
        self.initialInputSize = initialInputSize
        self.layers = []
    def addLayer(self, layerSize: int, layerType: activationFunction, **kwargs) -> None:
        if not 'fromDict' in kwargs:
            self.layers.append(layer(self.initialInputSize if len(self.layers) == 0 else self.layers[-1]._get_units_len_(), layerSize, layerType))
        else:
            self.layers.append(layer(self.initialInputSize if len(self.layers) == 0 else self.layers[-1]._get_units_len_(), kwargs["fromDict"]["layerSize"], kwargs["fromDict"]["layerType"], fromDict=kwargs["fromDict"]))
        
    def fit(self, error: float, alpha: float):
        print(f"layers: {len(self.layers)}")
        for layer in self.layers:
            layer.fit(error, alpha)
        
    def run(self, inputs: tuple):
        result = inputs
        for i, layer in enumerate(self.layers):
            result = layer.run(result)
        
        return result
    def saveAsJson(self, path):
        data = {"initialinputSize": self.initialInputSize,
                "layers": []}
        for layer in self.layers:
            data["layers"].append({"neurons": [],
                                   "layerSize": layer._get_units_len_(),
                                   "layerType": layer.activationFunctionType.value})
            for unit in layer.units:
                data["layers"][-1]["neurons"].append({"weight": unit.weights})
        with open(path, 'w') as f:
            json.dump(data, f, cls=CustomEncoder)

    def makeFromDict(self, dict):
        self.initialInputSize = dict["initialinputSize"]
        for layer in dict["layers"]:
            self.addLayer(layerSize=1, layerType=activationFunction.SIGMOID, fromDict=layer)