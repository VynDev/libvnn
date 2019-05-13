# libvnn (Vyn Neural Network)

This is a neural network library made in C++, i created it to learn about neural networks and data science so it may contains errors and/or not being compatible with you OS/Version

## Installation

### Requirements

- Linux (you should be able to do it on windows but with some modifications to the makefile)
- g++
- make

### Compilation

You can compile it as a static or a shared library:
- **make static** to get the static version (libvnn.a)
- **make dynamic** to get the shared version (libvnn.so)

### Use it in your project

The generated library (libvnn.a/libvnn.so) is in the **lib** folder  
The includes are in the **include** folder, you should just need to include **"Vyn/NeuralNetwork.h"**  

## Quick start

Learn how to use it with some examples and explanations

### Solving XOR

Let's see the C++ code for solving the XOR problem
```cpp
/* main.cpp */
#include <iostream>
#include <vector>
#include "Vyn/NeuralNetwork.h"

namespace NNet = Vyn::NeuralNetwork;

int main(void)
{
  srand(time(NULL));
  NNet::Network network;

  network.AddLayer(2, NNet::Activation::None);
  network.AddLayer(2, NNet::Activation::Sigmoid);
  network.AddLayer(1, NNet::Activation::Sigmoid);
  network.SetLearningRate(0.1);
  network.SetCostFunction(NNet::Cost::MSE);

  std::vector<NNet::Values> inputs =	{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  std::vector<NNet::Values> outputs =	{{0}, 	 {1},	 {1},	 {0}};

  for (int n = 0; n < 20000; ++n) // 20000 iterations
  {
    for (int i = 0; i < inputs.size(); ++i) // Each iteration, train each input
    {
      network.Predict(inputs[i]);
      network.Propagate(outputs[i]);
    }
  }
  
  std::cout << network.Predict(inputs[0])[0] << std::endl;
  std::cout << network.Predict(inputs[1])[0] << std::endl;
  std::cout << network.Predict(inputs[2])[0] << std::endl;
  std::cout << network.Predict(inputs[3])[0] << std::endl;
  return (0);
}
```
Compile and execute this code:  
```
g++ -I path_to_libvnn/include -L path_to_libvnn/lib main.cpp -o xor -lvnn
./xor
```

### Code explanation

Let's start from the beginning
```cpp
#include <iostream>
#include <vector>
#include "Vyn/NeuralNetwork.h"

namespace NNet = Vyn::NeuralNetwork;
```
I'll pass on \<iostream\> and \<vector\>.  
We include **Vyn/NeuralNetwork.h** for neural networks, all the functions are under the namespace **Vyn::NeuralNetwork** so we reduce it to **NNet** to make it easier.
```cpp
srand(time(NULL));
```
We need to initialize the random number generator ourself to get random weights in the neural network.
```cpp
NNet::Network network;

network.AddLayer(2, NNet::Activation::None);
network.AddLayer(2, NNet::Activation::Sigmoid);
network.AddLayer(1, NNet::Activation::Sigmoid);
```
We created the **network** object and then added layers to it.  
The **AddLayer** method create and add a new layer in the **network**, it take 2 arguments: the number of neurons in the layer and the activation function they will use. The first layer doesn't need an activation function.  
(at the moment, only Sigmoid and Softmax function are available)
```cpp
network.SetLearningRate(0.1);
network.SetCostFunction(NNet::Cost::MSE);
```
Because we will train our network model, we set the learning rate to 0.1 and we choose the Mean Squared Error (MSE) cost function
```cpp
std::vector<NNet::Values> inputs =	{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
std::vector<NNet::Values> outputs =	{{0}, 	 {1},	   {1},	   {0}};
```
For training our model, we need data, so we create them. **Values** is just a *std::vector* of *double*  
This 2 vectors are the inputs and outputs of the XOR gate:  
[0, 0] = 0  
[0, 1] = 1  
[1, 0] = 1  
[1, 1] = 0  
Now, let's train our model:
```cpp
for (int n = 0; n < 20000; ++n) // 20000 iterations
{
  for (int i = 0; i < inputs.size(); ++i) // Each iteration, train each input
  {
    network.Predict(inputs[i]);
    network.Propagate(outputs[i]);
  }
}
```
The **Predict** method takes a **Values** (vector of double) as inputs and return a **Values** as outputs  
The **Propagate** method takes a **Values** that corresponds to the expected outputs. A call to **Propagate** will always take the last output values from a call to **Predict** to calculate the error/cost

```cpp
std::cout << network.Predict(inputs[0])[0] << std::endl;
std::cout << network.Predict(inputs[1])[0] << std::endl;
std::cout << network.Predict(inputs[2])[0] << std::endl;
std::cout << network.Predict(inputs[3])[0] << std::endl;
```
Here we just print what our neural network model has learned. Note that **network.Predict(inputs[0])** returns a **Values** and we only have 1 ouput value, this explain why we append **[0]**

### Solving XOR (another method)

This is the same code, except for the training, we'll use the **Fit** builtin function
```cpp
int main(void)
{
  /* We need to initialize the random number generator ourself to get random weights in the neural network. */
  srand(time(NULL));
  /* We create the Network object */
  NNet::Network network;

  /* Adding layers to the network */
  network.AddLayer(2, NNet::Activation::None);
  network.AddLayer(2, NNet::Activation::Sigmoid);
  network.AddLayer(1, NNet::Activation::Sigmoid);
  /* Setting learning rate and cost function (Mean Squared Error) */
  network.SetLearningRate(0.1);
  network.SetCostFunction(NNet::Cost::MSE);

  /* Defining inputs and outputs */
  std::vector<NNet::Values> inputs =	{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  std::vector<NNet::Values> outputs =	{{0}, 	 {1},	 {1},	 {0}};

  /* Training */
  NNet::TrainingParameters parameters;
  parameters.trainingSetInputs = inputs;
  parameters.trainingSetOutputs = outputs;
  network.Fit(parameters, 1, 20000);
  
  /* Printing results */
  std::cout << network.Predict(inputs[0])[0] << std::endl;
  std::cout << network.Predict(inputs[1])[0] << std::endl;
  std::cout << network.Predict(inputs[2])[0] << std::endl;
  std::cout << network.Predict(inputs[3])[0] << std::endl;
  return (0);
}
```
The **Fit** method takes 3 arguments:
- The **TrainingParameters** object which contains the inputs dataset and their corresponding outputs  
- The batch size  
- The number of epoch/iteration  
This function will support some interesting features in the future

## Examples of projects using libvnn

### CarGenAI
Cars learning to drive in Unreal Engine 4 using genetic algorithm  
Demo: https://youtu.be/aslTSS2VpCA  
Github: https://github.com/VynOffline/CarGenAI

### Kaggle's Titanic challenge
Challenge overview: https://www.kaggle.com/c/titanic/overview  
Github: https://github.com/VynOffline/Kaggle-Titanic

## Documentation

There is no documentation available for the moment because the lib is still experimental/in development.

## What's coming next ?

- Support for Convolutional Neural Networks
- Support for CPU Multithreading
- More error handling
- Portability
