
# CONNECT-NN
Connect NN - An ANN (Artificial Neural Network) for word indentification.

# Making a Artificial Neural Network
First, we have to understand how the ANN works:

```mermaid
%%{init: {"flowchart": {"defaultRenderer": "elk"}} }%%
	graph LR
	
    A((Input 1)) --> C((Neuron))
    B((Input 2)) --> C((Neuron))
    
    A((Input 1)) --> D((Neuron))
    B((Input 2)) --> D((Neuron))
    
    A((Input 1)) --> E((Neuron))
    B((Input 2)) --> E((Neuron))
	
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

	C & D & E --> F & G & H --> I & J & K
	end

	O1((Output 1))
	O2((Output 2))
	I & J & K --> O1 & O2
```
This graph can be very confusing, here is a "simplification" of this diagram:

```mermaid
	graph LR
		In((Input))
		Ne{Neuron}
		Out((Output))
		In --> Ne --> Out
```
The neuron “thinks” because it carries out a process called **dot**, which multiplies the values ​​in the tables and adds them together. For example, I have two arrays: [1,2,3] and [3,2,1], so I want to apply the **dot** method, the result will be 10, because (1 * 3) + (2 * 2) + (3 * 1) = (3) + (4) + (3) = 3 + 4 + 3 = 10. For dot works, we need to multiply the neuron's inputs by the weight, that is the "memory" of the neuron. We need to use the **dot** method with this two arrays, and we will have the **linear function's output**.

To obtain the final output, the output of the linear function will go through an **activation function**, which will transform it into a readable output, and finaly the ***neuron's output***.

## Weights

For data processing, ANN neurons must have a weight for each input, to apply the dot (which I showed above). The weight must be random, for more efficiency and diverse outputs. See it in the diagram below:

```mermaid
%%{init: {"flowchart": {"defaultRenderer": "elk"}} }%%
	flowchart LR
		In((Input 1))
		In2((Input 2))
		subgraph Neuron
		Sum("∑")
		Ac(("f(x)"))
		We{Weight}
		end
		Out((Output))
		In & In2 & We --> Sum --> Ac --> Out
```

The weight is proportional to the dimension of the inputs, so if you give the AI the array [1,2,3], the weight should have three numbers inside it, such as [1,7756342, 2,837438758, -1,3893475637].
