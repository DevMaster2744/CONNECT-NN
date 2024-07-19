import numpy as np
from random import uniform
from enum import Enum


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


class unit():
    def __init__(self, inputSize: int, activationFunctionType: activationFunction) -> None:
        self.heights = [uniform(-1,1) for _ in range(inputSize)]
        self.activationFunctionType = activationFunctionType
        self.inputs = []
        self.output = 0
    def run(self, inputs: tuple) -> float:
        self.inputs = inputs

        result = np.dot(inputs, self.heights)

        result = activationFunctionFromEnum(result, self.activationFunctionType)
        return result

    def fit(self, error: float, alpha: float):
        for i in range(len(self.heights)):
            self.heights[i] +=  activationFunctionFromEnum(self.inputs[i], self.activationFunctionType) * error * alpha
    '''def setInputSize(self, inputSize: int):
        self.inp'''


class layer():
    def __init__(self, inputSize: int, layerSize: int, activationFunctionType: activationFunction) -> None:
        self.units = [unit(inputSize=inputSize, activationFunctionType=activationFunctionType) for _ in range(layerSize)]
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
    def addLayer(self, layerSize: int, layerType: activationFunction) -> None:
        if len(self.layers) == 0:
            self.layers.append(layer(self.initialInputSize, layerSize, layerType))
        else:
            self.layers.append(layer(self.layers[-1]._get_units_len_(), layerSize, layerType))
    def fit(self, error: float, alpha: float):
        for layer in self.layers:
            layer.fit(error, alpha)
    def run(self, inputs: tuple):
        result = inputs
        for i, layer in enumerate(self.layers):
            result = layer.run(result)
        
        return result

    