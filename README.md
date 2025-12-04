MiniGrad: Neural Network Engine from Scratch

MiniGrad is a lightweight, educational neural network library built entirely from scratch in Python. It implements a scalar-value autograd engine (automatic differentiation) and a PyTorch-like API for building and training neural networks.

This project is designed to demystify deep learning by exposing the internal mechanics of backpropagation, computational graphs, and gradient descent without relying on external heavy frameworks.

🌟 Key Features

Autograd Engine: Tracks operations on scalar values to build a dynamic computational graph (DAG).

Automatic Differentiation: Implements reverse-mode automatic differentiation (backpropagation) using the Chain Rule.

PyTorch-like API: Familiar structure using .backward(), .zero_grad()-style resets, and parameter updates.

Graph Visualization: Built-in integration with graphviz to visualize the forward pass and gradients.

Neural Network Modules: Includes Neuron, Layer, and MLP (Multi-Layer Perceptron) classes.

📂 Code Structure & Detailed Explanation

1. The Value Class (The Autograd Engine)

The core of the library is the Value class. It wraps standard Python numbers (scalars) to enable gradient tracking.

Initialization (__init__)

def __init__(self, data, _children=(), _op='', label=''):
    self.data = data        # The actual scalar value (e.g., 2.0)
    self.grad = 0.0         # Derivative of Loss with respect to this value
    self._backward = lambda: None  # Function to propagate gradients backward
    self._prev = set(_children)    # Pointers to parent nodes (creates the DAG)
    self._op = _op          # The operation that created this node (for debug/viz)


self.grad: Initially 0. It accumulates the derivative during backpropagation.

self._prev: This is crucial. It stores the links to the values that created this value, effectively building a Directed Acyclic Graph (DAG).

Operator Overloading (__add__, __mul__, etc.)

To allow expressions like a + b or a * b, Python's magic methods are overridden.

Forward Pass: Calculates the result (e.g., self.data + other.data).

Backward Pass (_backward closure): This is the heart of the engine. For every operation, we define a local function that knows how to calculate the gradient for its inputs using the Chain Rule.

Example: Addition

def _backward():
    # Local derivative of addition is 1.0
    # We accumulate (+=) gradients to handle cases where a variable is used multiple times
    self.grad += 1.0 * out.grad
    other.grad += 1.0 * out.grad


Activation Functions (tanh, exp)

Non-linearity is introduced here.

tanh: Squashes inputs between -1 and 1.

Derivative: $\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)$.

The _backward function for tanh implements this formula to pass gradients backward.

backward() (Topological Sort)

To calculate gradients for the whole network, we must process nodes in the correct order (from output back to input).

Topological Sort: A recursive algorithm visits every node in the graph, ensuring a node is processed only after all its dependencies are processed.

Reverse Iteration: We iterate through the sorted list in reverse and call _backward() on each node.

2. Neural Network Modules

Modeled after torch.nn, these classes abstract away the individual Value operations.

Neuron

Represents a single neuron: $y = \tanh(\sum(w_i \cdot x_i) + b)$.

Initializes random weights (self.w) and bias (self.b).

__call__: Performs the dot product of inputs and weights, adds bias, and applies non-linearity.

parameters(): Returns a list of [weights, bias] for optimization.

Layer

A collection of Neurons.

__init__: Creates a list of nout neurons.

__call__: Passes the input to every neuron and returns their outputs.

MLP (Multi-Layer Perceptron)

A full feed-forward neural network.

__init__: Takes a list of layer sizes (e.g., [3, 4, 4, 1]) and chains Layer objects together.

__call__: Sequentially passes data through each layer.

3. Graph Visualization (draw_dot)

The code includes helper functions (trace, draw_dot) using the graphviz library.

Purpose: Visualizes the computational graph.

Nodes: Represent Value objects (showing data and grad).

Edges: Represent the flow of data (operations).

Usage: Extremely useful for debugging to see if the graph is connected correctly.

🏃 Example Usage & Training Loop

Here is how you can use MiniGrad to train a simple network:

# 1. Define the Network
# 3 inputs, two hidden layers of 4 neurons, 1 output
n = MLP(3, [4, 4, 1])

# 2. Define Inputs and Targets
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets

# 3. Training Loop
for k in range(20):
  
  # a. Forward Pass
  ypred = [n(x) for x in xs]
  loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
  
  # b. Zero Gradients
  # Vital! Otherwise grads accumulate from previous steps
  for p in n.parameters():
    p.grad = 0.0
  
  # c. Backward Pass
  loss.backward()
  
  # d. Update (Gradient Descent)
  for p in n.parameters():
    p.data += -0.1 * p.grad
  
  print(f"Step {k} | Loss: {loss.data}")


⚠️ The "Accumulation" Bug Explanation

The code correctly handles a common pitfall in backpropagation implementation:

Wrong Way: self.grad = ... (Overwrites the gradient).

Correct Way: self.grad += ... (Accumulates the gradient).

Why?
If a variable is used multiple times (e.g., b = a + a), the gradient must flow back from both branches. The multivariate chain rule states these gradients should be summed. Using += ensures we capture the total influence of a variable on the loss.

✅ Correctness Verification

The library includes test functions (test_sanity_check, test_more_ops) that compare MiniGrad results against PyTorch to ensure mathematical accuracy.

Checks: Forward pass values and Backward pass gradients.

Result: Asserts that both libraries produce identical results within a tolerance.

🛠 Prerequisites

To run this code, you need:

Python 3.x

graphviz (for visualization)

numpy (optional, used for some helper math)

torch (optional, only used in the sanity check tests)

pip install graphviz numpy torch
