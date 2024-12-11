import math
from random import random

class Neuron():
    def __init__(self, input_size, output_size = 1):
        self.weights = [random() for i in range(0,input_size)]
        self.input_size = input_size
        self.ouput_size = output_size
        self.bias = random()
        
    def calc(self, input_values):
        out_value = sum([self.weights[i]+value for i, value in enumerate(input_values)])+self.bias
        out_value = sigmoid(out_value)
        return out_value

def sigmoid(value):
    value = 1/(1+math.e**(-value))
    return value

class Layer():
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.ouput_size = output_size
        self.neurons = [Neuron(input_size) for i in range(0,output_size)]

    def calc_output(self, input_values):
        self.out_values = [neuron.calc(input_values) for neuron in self.neurons]

class Model():
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []

    def add_layer(self, layer_size):
        if len(self.layers) > 1: layer_input = self.layers[-1].output_size
        else: layer_input = self.input_size
        layer = Layer(layer_input,layer_size)
        self.layers.append(layer)

    def predict(self):
        pass