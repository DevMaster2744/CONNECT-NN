
# CONNECT-NN
Connect NN - A simple Python module designed to be used in the development of Artificial Neural Networks

# Development
Connect NN can make custom neural networks, and basic activation functions that can be used in deep learning
 applications

# Making a Artificial Neural Network
First, we have to understand how the ANN works:

```mermaid
	flowchart LR
	
    A((Input 1)) --> C((Neuron))
    B((Input 2)) --> C((Neuron))
    
    A((Input 1)) --> D((Neuron))
    B((Input 2)) --> D((Neuron))
    
    A((Input 1)) --> E((Neuron))
    B((Input 2)) --> E((Neuron))
    
    C & D & E ---> F & G & H ---> I & J & K
	
	subgraph Hidden Layer
	C((Neuron))
	D((Neuron))
	E((Neuron))
	
	F((Neuron))
	G((Neuron))
	H((Neuron))
	
	I((Neuron))
	J((Neuron))
	K((Neuron))
	end

	O1((Output 1))
	O2((Output 2))

	I & J & K --> O1 & O2
    
    

