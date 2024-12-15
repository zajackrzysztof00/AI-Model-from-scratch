"""
This script implements a basic feedforward neural network with backpropagation 
for training. The network supports multiple layers, with each layer consisting of 
neurons that apply the sigmoid activation function. The training process uses 
gradient descent to adjust the weights and biases of the neurons based on the 
error (cost) between the predicted and desired outputs.

Key Features:
- Forward propagation to compute outputs
- Backpropagation to calculate gradients
- Gradient descent for weight and bias updates
- Modular design with Neuron, Layer, and Model classes

Dependencies:
- Python standard libraries: math, random, and typing

Usage:
1. Create a model specifying the input size.
2. Add layers with the desired number of neurons.
3. Train the model using input data and expected output values.

"""

import math
from random import random
from typing import List

# Activation Functions and Derivatives
def sigmoid(value: float) -> float:
    """
    Computes the sigmoid activation function.

    Args:
        value (float): The input value.

    Returns:
        float: The result of the sigmoid function.
    """
    try:
        return 1 / (1 + math.exp(-value))
    except OverflowError:
        return 0.0


def derivative_sigmoid(value: float) -> float:
    """
    Computes the derivative of the sigmoid function.

    Args:
        value (float): The input value.

    Returns:
        float: The derivative of the sigmoid function.
    """
    sig = sigmoid(value)
    return sig * (1 - sig)


def calc_z_value(weights: List[float], values: List[float], bias: float) -> float:
    """
    Computes the weighted sum (z-value) for a neuron.

    Args:
        weights (List[float]): Weights of the neuron.
        values (List[float]): Input values to the neuron.
        bias (float): Bias of the neuron.

    Returns:
        float: The z-value of the neuron.
    """
    return sum(w * v for w, v in zip(weights, values)) + bias


def derivative_cost_to_output(output_value: float, desired_value: float) -> float:
    """
    Computes the derivative of the cost function with respect to the output.

    Args:
        output_value (float): The predicted output value.
        desired_value (float): The desired output value.

    Returns:
        float: The derivative of the cost.
    """
    return 2 * (output_value - desired_value)


# Neuron Class
class Neuron:
    """
    Represents a single neuron in the neural network.
    """
    def __init__(self, input_size: int):
        """
        Initializes a neuron with random weights and bias.

        Args:
            input_size (int): The number of inputs to the neuron.
        """
        self.weights: List[float] = [random() for _ in range(input_size)]
        self.bias: float = random()
        self.weight_cost: List[float] = [0.0 for _ in range(input_size)]
        self.constant_part: float = 0.0

    def calc(self, input_values: List[float]) -> float:
        """
        Computes the output of the neuron.

        Args:
            input_values (List[float]): Input values to the neuron.

        Returns:
            float: The output of the neuron after applying the sigmoid function.
        """
        self.z_value = calc_z_value(self.weights, input_values, self.bias)
        self.out_value = sigmoid(self.z_value)
        return self.out_value

    def remember_weight_cost(self, weight_idx: int, weight_cost: float):
        """
        Updates the weight cost for the specified weight index.

        Args:
            weight_idx (int): The index of the weight.
            weight_cost (float): The calculated weight cost.
        """
        self.weight_cost[weight_idx] += weight_cost

    def remember_constant_part(self, constant: float):
        """
        Stores the constant part of the gradient for bias updates.

        Args:
            constant (float): The gradient constant part.
        """
        self.constant_part = constant

    def modify_weights(self):
        """
        Updates the weights based on the accumulated weight costs.
        """
        self.weights = [w - wc for w, wc in zip(self.weights, self.weight_cost)]
        self.weight_cost = [0.0 for _ in range(len(self.weights))]  # Reset weight costs after update

    def modify_bias(self):
        """
        Updates the bias based on the stored constant part.
        """
        self.bias -= self.constant_part


# Layer Class
class Layer:
    """
    Represents a single layer in the neural network.
    """
    def __init__(self, input_size: int, output_size: int):
        """
        Initializes a layer with a specified number of neurons.

        Args:
            input_size (int): The number of inputs to the layer.
            output_size (int): The number of neurons in the layer.
        """
        self.neurons: List[Neuron] = [Neuron(input_size) for _ in range(output_size)]
        self.out_values: List[float] = []

    def calc_output(self, input_values: List[float]) -> None:
        """
        Computes the output values of all neurons in the layer.

        Args:
            input_values (List[float]): Input values to the layer.
        """
        self.out_values = [neuron.calc(input_values) for neuron in self.neurons]


# Model Class
class Model:
    """
    Represents the neural network model.
    """
    def __init__(self, input_size: int):
        """
        Initializes the model with an input size.

        Args:
            input_size (int): The number of inputs to the model.
        """
        self.input_size = input_size
        self.layers: List[Layer] = []

    def add_layer(self, layer_size: int) -> None:
        """
        Adds a new layer to the model.

        Args:
            layer_size (int): The number of neurons in the new layer.
        """
        layer_input = len(self.layers[-1].neurons) if self.layers else self.input_size
        self.layers.append(Layer(layer_input, layer_size))

    def predict(self, input_values: List[float]) -> List[float]:
        """
        Makes a prediction using the model.

        Args:
            input_values (List[float]): Input data for prediction.

        Returns:
            List[float]: The predicted output values.
        """
        values = input_values
        for layer in self.layers:
            layer.calc_output(values)
            values = layer.out_values
        return values

    def train(self, train_values: List[float], train_result: List[float]) -> None:
        """
        Trains the model using a single training example.

        Args:
            train_values (List[float]): Input training data.
            train_result (List[float]): Desired output values.
        """
        # Forward pass
        self.predict(train_values)

        # Backpropagation
        for layer_idx in reversed(range(len(self.layers))):
            layer = self.layers[layer_idx]
            for neuron_idx, neuron in enumerate(layer.neurons):
                previous_values = self.layers[layer_idx - 1].out_values if layer_idx > 0 else train_values
                z_value = neuron.z_value

                if layer_idx == len(self.layers) - 1:  # Output layer
                    constant_part = derivative_sigmoid(z_value) * derivative_cost_to_output(
                        neuron.out_value, train_result[neuron_idx]
                    )
                else:  # Hidden layers
                    next_layer = self.layers[layer_idx + 1]
                    constant_part = derivative_sigmoid(z_value) * sum(
                        next_neuron.weights[neuron_idx] * next_neuron.constant_part
                        for next_neuron in next_layer.neurons
                    )

                # Store the constant part for the current neuron
                neuron.remember_constant_part(constant_part)

                # Update weight costs based on outputs of the current neuron
                for weight_idx in range(len(neuron.weights)):
                    weight_cost = previous_values[weight_idx] * constant_part
                    neuron.remember_weight_cost(weight_idx, weight_cost)

        # Update weights and biases
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.modify_weights()
                neuron.modify_bias()


# Model Configuration
model = Model(128)
model.add_layer(10)
model.add_layer(8)
model.add_layer(6)
model.add_layer(4)

# Training Configuration
input_val = [i for i in range(128)]
desired_output = [0.1, 0.2, 0.3, 0.4]

# Training Loop
for count in range(1000):  # Train for 1000 iterations
    prediction = model.predict(input_val)
    print(f"Iteration {count + 1}: Prediction {prediction}, Desired {desired_output}")
    model.train(input_val, desired_output)
