import math
from random import random

def sigmoid(value):
    value = 1/(1+math.e**(-value))
    return value

def derivative_sigmoid(value):
    value = sigmoid(value)*(1-sigmoid(value))
    return value

def calc_z_value(weights, values, bias):
    z_value = sum([weights[i]*value for i, value in enumerate(values)])+bias
    return z_value

def derivative_cost_to_output(output_value, disired_value):
    cost = 2*(output_value-disired_value)
    return cost

def derivatives_cost_to_weight_retio(before_values, output_value, disired_value, weights, bias):
    retio = before_values*derivative_sigmoid(calc_z_value(weights, before_values, bias))*derivative_cost_to_output(output_value, disired_value)
    return retio

def derivatives_cost_to_bias_retio(before_value, output_value, disired_value, weight, bias):
    retio = derivative_sigmoid(calc_z_value(weight, before_value, bias))*derivative_cost_to_output(output_value, disired_value)
    return retio

def calc_average(values):
    return sum(values)/len(values)

def calc_sum_der_of_cost_to_output(outputs, disired_outputs):
    cost_sum = sum(derivative_cost_to_output(output, disired_outputs[i]) for i, output in enumerate(outputs))
    return cost_sum
        
class Neuron():
    def __init__(self, input_size, output_size = 1):
        self.weights = [random() for i in range(0,input_size)]
        self.input_size = input_size
        self.ouput_size = output_size
        self.bias = random()
        
    def calc(self, input_values):
        out_value = sum([self.weights[i]*value for i, value in enumerate(input_values)])+self.bias
        out_value = sigmoid(out_value)
        return out_value

class Layer():
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.neurons = [Neuron(input_size) for i in range(0,output_size)]

    def calc_output(self, input_values):
        self.out_values = [neuron.calc(input_values) for neuron in self.neurons]

    def calc_cost(self, train_result):
        self.cost = [(self.out_values[i] - train_result[i])**2 for i in range(0,len(self.out_values))]

class Model():
    def __init__(self, input_size):
        self.input_size = input_size
        self.layers = []
        self.add_layer(input_size)

    def add_layer(self, layer_size):
        if len(self.layers) > 1: layer_input = self.layers[-1].output_size
        else: layer_input = self.input_size
        layer = Layer(layer_input,layer_size)
        self.layers.append(layer)

    def predict(self, input_values):
        values = input_values
        for layer in self.layers:
            layer.calc_output(values)
            values = layer.out_values
        return values
    
    def train(self, train_values, train_result):
        self.predict(train_values)
        for i, layer in enumerate(self.layers):
            self.layers[-i].calc_cost(train_result)
