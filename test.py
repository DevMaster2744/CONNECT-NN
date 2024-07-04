import connectnn as cn
from time import sleep
import random

nn = cn.NeuralNetwork(2)

nn.addLayer(35, cn.activationFunction.TANH)

nn.addLayer(40, cn.activationFunction.TANH)

nn.addLayer(5, cn.activationFunction.TANH)

nn.addLayer(1, cn.activationFunction.SIGMOID)