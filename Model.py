import math
from random import random

def sigmoid(value):
    try:
        value = 1/(1+math.e**(-value))
    except:
        value = 0
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

class Neuron():
    def __init__(self, input_size, output_size = 1):
        self.weights = [random() for i in range(0,input_size)]
        self.input_size = input_size
        self.ouput_size = output_size
        self.bias = random()
        self.weight_cost = []
        self.constant_part = []
        
    def calc(self, input_values):
        self.out_value = sum([self.weights[i]*value for i, value in enumerate(input_values)])+self.bias
        self.out_value = sigmoid(self.out_value)
        return self.out_value
    
    def remember_weight_cost(self, weight_cost):
        self.weight_cost.append(weight_cost)

    def remember_cosnstant_part(self, constant):
        self.constant_part.append(constant)

    def remember_z_value(self, z_value):
        self.z_value = z_value

    def modify_weights(self):
        new_weights = []
        for i, weight in enumerate(self.weights):
            new_weights.append(weight-self.weight_cost[i])
        self.weights = new_weights

    def remember_bias_change(self, change):
        self.bias_change = change
    
    def modify_bias(self):
        self.bias = self.bias - self.bias_change

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

model = Model(8)
model.add_layer(8)
model.add_layer(8)
model.add_layer(8)
model.add_layer(8)
input_val = [i for i in range(0,8)]
#dis_out = [(1+i)/5 for i in range(0,4)]
dis_out = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
count = 0
while True:
    count = count+1
    prediction = model.predict(input_val)
    print(prediction)
    print(dis_out)
    #print(model.layers)
    l_num = len(model.layers)
    for layer in reversed(model.layers):
        #print(layer.out_values)
        l_num = l_num - 1
        for n_num , neuron in enumerate(layer.neurons):
            #print(neuron)
            if l_num-1 >= 0:
                previous_values =  model.layers[l_num-1].out_values
            else:
                previous_values =  input_val
            z_value = calc_z_value(neuron.weights, previous_values, neuron.bias)
            neuron.remember_z_value(z_value)
            for w_num, weight in enumerate(neuron.weights):
                if l_num == len(model.layers)-1:
                    constant_part = derivative_sigmoid(z_value)*derivative_cost_to_output(neuron.out_value, dis_out[n_num])
                else:
                    constant_part = derivative_sigmoid(z_value)*sum([w*
                                                                    derivative_sigmoid(model.layers[l_num+1].neurons[n_num].z_value)*
                                                                    model.layers[l_num+1].neurons[n_num].constant_part[i] 
                                                                    for i, w in enumerate(neuron.weights)])
                neuron.remember_cosnstant_part(constant_part)
                #print(w_num)
                weight_cost_ratio = previous_values[w_num]*constant_part
                #print(weight_cost_ratio)
                neuron.remember_weight_cost(weight_cost_ratio)
    l_num = len(model.layers)
    for lauer in reversed(model.layers):
        l_num = l_num - 1
        for n_num, neuron in enumerate(lauer.neurons):
            if l_num-1 >= 0:
                previous_values =  model.layers[l_num-1].out_values
            else:
                previous_values =  input_val
            z_value = neuron.z_value
            if l_num == len(model.layers)-1:
                    constant_part = derivative_sigmoid(z_value)*derivative_cost_to_output(neuron.out_value, dis_out[n_num])
            else:
                constant_part = derivative_sigmoid(z_value)*sum([w*
                                                                derivative_sigmoid(model.layers[l_num+1].neurons[n_num].z_value)*
                                                                model.layers[l_num+1].neurons[n_num].constant_part[i] 
                                                                for i, w in enumerate(neuron.weights)])
            bias_change = constant_part
            neuron.remember_bias_change(bias_change)

    for layer in  reversed(model.layers):
        for n_num, neuron in enumerate(layer.neurons):
            neuron.modify_weights()
            neuron.modify_bias()
