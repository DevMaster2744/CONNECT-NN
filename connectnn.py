import numpy as np
from random import uniform
from enum import Enum


class activationFunction(Enum):
    SIGMOID = 1
    BINARY = 2
    TANH = 3

def activationFunctionFromEnum(x: float, activationFunctionType: activationFunction):
    if activationFunctionType == activationFunction.SIGMOID:
        return 1 / (1 + np.exp(-x))
    
    if activationFunctionType == activationFunction.TANH:
        th = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return th
    
    if activationFunctionType == activationFunction.BINARY:
        return 1 if (1 / (1 + np.exp(-x)) > 0.5) else 0


class unit():
    def __init__(self, inputSize: int, activationFunctionType: activationFunction) -> None:
        
        self.heights = [uniform(-1,1) for _ in range(inputSize)]
        self.activationFunctionType = activationFunctionType

class layer():
    def __init__(self, inputSize: int, layerSize: int, activationFunctionType: activationFunction) -> None:
        self.units = [unit(inputSize=inputSize, activationFunctionType=activationFunctionType) for _ in range(layerSize)]


class NeuralNetwork():
    def __init__(self, initialInputSize: int) -> None:
        self.initialInputSize = initialInputSize
        self.layers = []

    def addLayer(self, layerSize: int, layerType: activationFunction):
        self.layers.append(layer(self.initialInputSize, layerSize, layerType))
        