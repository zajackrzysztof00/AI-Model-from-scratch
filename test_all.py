
"""
This script contains a set of unit tests for a simple neural network model. 
The tests cover basic functionality like prediction, weight and bias initialization, 
forward propagation, backpropagation, and training convergence. These tests ensure 
that the neural network behaves as expected under various conditions.

Tests include:
1. Weight and bias initialization verification.
2. Output size verification after prediction.
3. Forward propagation correctness.
4. Weight updates during training (backpropagation).
5. Training convergence and cost function behavior.
6. Model structure and input validation.

Dependencies:
- Python standard libraries: random
"""

from Model import Model
from random import random

def test_weight_and_bias_initialization() -> None:
    """
    Test if the weights and biases are initialized to random values for each neuron.

    Asserts:
        - Weights should be initialized to random floats.
        - Bias should be initialized to a random float.
    """
    model = Model(128)
    model.add_layer(10)
    model.add_layer(8)
    
    for layer_idx, layer in enumerate(model.layers):
        for neuron_idx, neuron in enumerate(layer.neurons):
            # Ensure weights have the correct size depending on the layer's input size
            assert len(neuron.weights) == (model.input_size if layer_idx == 0 else len(model.layers[layer_idx - 1].neurons))
            assert isinstance(neuron.bias, float)  # Bias is a float
            assert isinstance(neuron.weights[0], float)  # First weight is a float

def test_model_prediction_output_size() -> None:
    """
    Test if the model's prediction output size matches the last layer's size.

    Asserts:
        - The prediction size should match the output size of the last layer.
    """
    model = Model(16)
    model.add_layer(32)
    model.add_layer(16)
    prediction = model.predict([random() for _ in range(16)])
    
    assert len(prediction) == 16  # Output size should match the last layer size

def test_forward_propagation() -> None:
    """
    Test the forward propagation process and output correctness.
    
    Asserts:
        - The output size should match the output layer size.
        - Outputs should be between 0 and 1 (as they go through sigmoid activation).
    """
    model = Model(3)  # Example with 3 input neurons
    model.add_layer(5)  # Example with 5 neurons in the first hidden layer
    model.add_layer(2)  # Example with 2 neurons in the output layer
    
    # Inputs
    inputs = [0.1, 0.2, 0.3]
    outputs = model.predict(inputs)
    
    assert len(outputs) == 2  # Output layer size
    assert all(0 <= output <= 1 for output in outputs)  # Outputs should be between 0 and 1 (sigmoid function)

def test_backpropagation_and_weight_updates() -> None:
    """
    Test if backpropagation correctly updates the weights and biases after training.
    
    Asserts:
        - Weights and biases should change after training.
    """
    model = Model(3)
    model.add_layer(5)
    model.add_layer(2)
    
    initial_weights = [neuron.weights.copy() for layer in model.layers for neuron in layer.neurons]
    initial_biases = [neuron.bias for layer in model.layers for neuron in layer.neurons]
    
    # Train the model with an example
    inputs = [0.1, 0.2, 0.3]
    desired_output = [0.8, 0.9]
    model.train(inputs, desired_output)
    
    # Check if weights and biases were updated
    for idx, layer in enumerate(model.layers):
        for neuron_idx, neuron in enumerate(layer.neurons):
            assert neuron.weights != initial_weights[idx * len(layer.neurons) + neuron_idx]
            assert neuron.bias != initial_biases[idx * len(layer.neurons) + neuron_idx]

def test_training_convergence() -> None:
    """
    Test if the model's predictions change over multiple training iterations.
    
    Asserts:
        - Predictions should change over iterations indicating the model is learning.
    """
    model = Model(3)
    model.add_layer(5)
    model.add_layer(2)
    
    input_values = [0.1, 0.2, 0.3]
    desired_output = [0.8, 0.9]
    
    initial_predictions = model.predict(input_values)
    
    for i in range(100):  # Train for 100 iterations
        model.train(input_values, desired_output)
        
        # Check if predictions change between iterations
        new_predictions = model.predict(input_values)
        assert any(abs(a - b) > 1e-4 for a, b in zip(initial_predictions, new_predictions))
        initial_predictions = new_predictions

def test_model_with_multiple_layers() -> None:
    """
    Test the model with multiple layers and verify the number of neurons in each layer.
    
    Asserts:
        - The model should have the expected number of layers and neurons in each layer.
    """
    model = Model(10)
    model.add_layer(20)  # First hidden layer
    model.add_layer(30)  # Second hidden layer
    model.add_layer(10)  # Output layer
    
    assert len(model.layers) == 4  # Including input layer, should have 4 layers
    
    for i, layer in enumerate(model.layers[1:]):
        assert len(layer.neurons) == [20, 30, 10][i]  # Check if the layer has the correct number of neurons

def test_invalid_input_size() -> None:
    """
    Test if the model raises an error when given an input size that doesn't match the input layer.
    
    Asserts:
        - The model should raise an error for invalid input sizes.
    """
    model = Model(5)
    model.add_layer(10)
    
    try:
        model.predict([0.1, 0.2])  # Invalid input size (should be 5)
        assert False, "Model should have raised an error for invalid input size"
    except ValueError:
        pass  # Expected behavior

def test_cost_function_behavior() -> None:
    """
    Test if the cost function correctly decreases after training, based on the training target.
    
    Asserts:
        - The cost should decrease after training with far from and close to desired outputs.
    """
    model = Model(3)
    model.add_layer(2)
    
    inputs = [0.1, 0.2, 0.3]
    desired_output_far = [0.8, 0.9]
    desired_output_close = [0.1, 0.2]
    
    # Far from desired output
    prediction_far = model.predict(inputs)
    initial_cost_far = sum((p - d) ** 2 for p, d in zip(prediction_far, desired_output_far))
    
    model.train(inputs, desired_output_far)
    prediction_far_after = model.predict(inputs)
    cost_after_far = sum((p - d) ** 2 for p, d in zip(prediction_far_after, desired_output_far))
    
    assert cost_after_far < initial_cost_far  # Cost should reduce after training
    
    # Close to desired output
    prediction_close = model.predict(inputs)
    initial_cost_close = sum((p - d) ** 2 for p, d in zip(prediction_close, desired_output_close))
    
    model.train(inputs, desired_output_close)
    prediction_close_after = model.predict(inputs)
    cost_after_close = sum((p - d) ** 2 for p, d in zip(prediction_close_after, desired_output_close))
    
    assert cost_after_close < initial_cost_close  # Cost should reduce after training
