from Model import Neuron, Layer
from random import random

def test_neuron_calc_finction():
    test_neuron = Neuron(32)

    assert test_neuron.calc([random() for i in range(0,test_neuron.input_size)]) <= 1

def test_layers_output_size():
    test_layer = Layer(32,32)

    assert len(test_layer.neurons) == 32
    for test_neuron in test_layer.neurons:
        assert test_neuron.calc([random() for i in range(0,test_neuron.input_size)]) <= 1

def test_layer_output_calc():
    test_layer = Layer(32,32)
    inputs = [random() for i in range(0,test_layer.input_size)]
    test_layer.calc_output(inputs)

    assert len(test_layer.out_values) == test_layer.ouput_size