import connectnn as cn

nn = cn.NeuralNetwork(2)
nn.addLayer(5, cn.activationFunction.TANH)