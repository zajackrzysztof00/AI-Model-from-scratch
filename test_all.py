from Model import Neuron, Layer, Model
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

    assert len(test_layer.out_values) == test_layer.output_size

global model
model = Model(16)

def test_model_creation():
    model.add_layer(32)
    assert len(model.layers) == 2
    model.add_layer(16)
    assert len(model.layers) == 3
    test_layer = model.layers[-1]
    for test_neuron in test_layer.neurons:
        assert test_neuron.calc([random() for i in range(0,test_neuron.input_size)]) <= 1

def test_model_prediction():
    model.add_layer(32)
    model.add_layer(16)
    predictions = model.predict([i for i in range(0,16)])
    for prediction in predictions:
        assert prediction <= 1
    model.train([i for i in range(0,16)],[0 for i in range(0,32)]+[1])
    print(model.layers[-1].cost)
