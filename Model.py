# Neural Network Implementation
# This script implements a basic neural network from scratch in Python. It includes the following components:
# 1. **Activation Functions**: Implements the sigmoid function and its derivative.
# 2. **Neuron Class**: Represents a single neuron in the network, including weight and bias management.
# 3. **Layer Class**: Represents a layer of neurons in the network.
# 4. **Model Class**: Represents the overall neural network model with multiple layers and provides prediction and training functionalities.
# 5. **Training Loop**: Demonstrates a simple training loop using a predefined input and desired output.

import math
from random import random
from typing import List

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

class Neuron:
    """
    Represents a single neuron in the neural network.
    """
    def __init__(self, input_size: int):
        """
        Initializes a Neuron with random weights and bias.

        Args:
            input_size (int): The number of inputs to the neuron.
        """
        self.weights: List[float] = [random() for _ in range(input_size)]
        self.bias: float = random()
        self.weight_cost: List[float] = []
        self.constant_part: List[float] = []

    def calc(self, input_values: List[float]) -> float:
        """
        Computes the output of the neuron.

        Args:
            input_values (List[float]): Input values to the neuron.

        Returns:
            float: The output of the neuron after applying the sigmoid activation function.
        """
        self.out_value = calc_z_value(self.weights, input_values, self.bias)
        self.out_value = sigmoid(self.out_value)
        return self.out_value

    def remember_weight_cost(self, weight_cost: float):
        """
        Stores the weight cost for later use.

        Args:
            weight_cost (float): The weight cost to store.
        """
        self.weight_cost.append(weight_cost)

    def remember_constant_part(self, constant: float):
        """
        Stores the constant part for later use.

        Args:
            constant (float): The constant part to store.
        """
        self.constant_part.append(constant)

    def remember_z_value(self, z_value: float):
        """
        Stores the z-value for the neuron.

        Args:
            z_value (float): The z-value to store.
        """
        self.z_value = z_value

    def modify_weights(self):
        """
        Updates the weights based on the stored weight costs.
        """
        self.weights = [w - wc for w, wc in zip(self.weights, self.weight_cost)]

    def remember_bias_change(self, change: float):
        """
        Stores the bias change for later use.

        Args:
            change (float): The bias change to store.
        """
        self.bias_change = change

    def modify_bias(self):
        """
        Updates the bias based on the stored bias change.
        """
        self.bias -= self.bias_change

class Layer:
    """
    Represents a single layer in the neural network.
    """
    def __init__(self, input_size: int, output_size: int):
        """
        Initializes a Layer with a specified number of neurons.

        Args:
            input_size (int): The number of inputs to the layer.
            output_size (int): The number of neurons in the layer.
        """
        self.neurons: List[Neuron] = [Neuron(input_size) for _ in range(output_size)]

    def calc_output(self, input_values: List[float]) -> None:
        """
        Computes the output of the layer.

        Args:
            input_values (List[float]): Input values to the layer.
        """
        self.out_values: List[float] = [neuron.calc(input_values) for neuron in self.neurons]

    def calc_cost(self, train_result: List[float]) -> None:
        """
        Computes the cost for the layer based on the desired output.

        Args:
            train_result (List[float]): The desired output values for the layer.
        """
        self.cost: List[float] = [(out - train_result[i])**2 for i, out in enumerate(self.out_values)]

class Model:
    """
    Represents the neural network model.
    """
    def __init__(self, input_size: int):
        """
        Initializes the Model with an input size.

        Args:
            input_size (int): The number of inputs to the model.
        """
        self.input_size = input_size
        self.layers: List[Layer] = []

    def add_layer(self, layer_size: int) -> None:
        """
        Adds a new layer to the model.

        Args:
            layer_size (int): The size (number of neurons) of the layer to add.
        """
        layer_input = self.layers[-1].output_size if self.layers else self.input_size
        self.layers.append(Layer(layer_input, layer_size))

    def predict(self, input_values: List[float]) -> List[float]:
        """
        Makes a prediction using the current model.

        Args:
            input_values (List[float]): Input values for prediction.

        Returns:
            List[float]: Predicted output values.
        """
        values = input_values
        for layer in self.layers:
            layer.calc_output(values)
            values = layer.out_values
        return values

    def train(self, train_values: List[float], train_result: List[float]) -> None:
        """
        Trains the model on a single batch of data.

        Args:
            train_values (List[float]): Input training data.
            train_result (List[float]): Desired output training data.
        """
        self.predict(train_values)
        for i, layer in enumerate(self.layers):
            self.layers[-i].calc_cost(train_result)

# Model and Training Configuration
model = Model(8)
model.add_layer(8)
model.add_layer(8)
model.add_layer(8)
model.add_layer(8)

input_val = [i for i in range(8)]
dis_out = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# Training Loop
count = 0
while count < 1000:  # Limit training iterations
    count += 1
    prediction = model.predict(input_val)
    print(f"Iteration {count}: Prediction {prediction}, Desired {dis_out}")

    for layer_idx, layer in reversed(list(enumerate(model.layers))):
        for neuron_idx, neuron in enumerate(layer.neurons):
            previous_values = model.layers[layer_idx - 1].out_values if layer_idx > 0 else input_val

            z_value = calc_z_value(neuron.weights, previous_values, neuron.bias)
            neuron.remember_z_value(z_value)

            if layer_idx == len(model.layers) - 1:  # Output layer
                constant_part = derivative_sigmoid(z_value) * derivative_cost_to_output(neuron.out_value, dis_out[neuron_idx])
            else:  # Hidden layers
                next_layer = model.layers[layer_idx + 1]
                constant_part = derivative_sigmoid(z_value) * sum(
                    w * derivative_sigmoid(next_neuron.z_value) * next_neuron.constant_part[neuron_idx]
                    for w, next_neuron in zip(neuron.weights, next_layer.neurons)
                )

            neuron.remember_constant_part(constant_part)
            neuron.remember_bias_change(constant_part)

            for weight_idx, weight in enumerate(neuron.weights):
                weight_cost = previous_values[weight_idx] * constant_part
                neuron.remember_weight_cost(weight_cost)

    for layer in reversed(model.layers):
        for neuron in layer.neurons:
            neuron.modify_weights()
            neuron.modify_bias()
