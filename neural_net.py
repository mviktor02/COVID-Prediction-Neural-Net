from os import mkdir
from os.path import join, exists
from datetime import datetime
from numpy import ndarray
from sklearn.metrics import mean_squared_error

import json
import time
import numpy as np

from activation import Activation

HIST = 7


def __get_train_data(data: list, percent: float):
    divider = int(len(data)*percent)
    return data[:divider]


def __get_test_data(data: list, percent: float):
    divider = int(len(data)*percent)
    return data[divider:]


def apply_simple(data: list, hidden_layers: list, activation: Activation, acceptable_error_margin=0.0001, save=False, bias=1, train_percent=0.6, load_file=None) -> ndarray:
    """
    :param data: latest covid data; daily new cases and new deaths
    :param hidden_layers: number of neurons in each hidden layer
    :param activation: activation function to be used in each layer
    :param train_percent: percentage of covid data to be used for training; rest will be used for testing
    :param acceptable_error_margin: margin of error where the neural network should stop training
    :param save: should the trained network be exported to a json file?
    :param bias: bias of the network
    :param load_file: trained network data to load; if given, save is ignored and this will be used as the trained data
    :return: list of predicted values; [new_cases, new_deaths]
    """
    activations = [activation for _ in range(len(hidden_layers)+3)]
    return apply(data, hidden_layers, activations, train_percent, acceptable_error_margin, save, bias, load_file)


def apply(data: list, hidden_layers: list, activations: list, train_percent=0.6, acceptable_error_margin=0.0001, save=False, bias=1, load_file=None) -> ndarray:
    """
    :param data: latest covid data; daily new cases and new deaths
    :param hidden_layers: number of neurons in each hidden layer
    :param activations: list of activation functions to use in each layer; has to have a length of len(hidden_layers)+3
    :param train_percent: percentage of covid data to be used for training; rest will be used for testing
    :param acceptable_error_margin: margin of error where the neural network should stop training
    :param save: should the trained network be exported to a json file?
    :param bias: bias of the network
    :param load_file: trained network data to load; if given, save is ignored and this will be used as the trained data
    :return: list of predicted values; [new_cases, new_deaths]
    """
    if load_file is None:
        train_result = __train(data, activations, hidden_layers, bias, train_percent, acceptable_error_margin)
        if save:
            __save(train_result)
    else:
        train_result = __load(load_file)

    test_result = __test(data, activations, train_result[0], train_result[1], bias, 1-train_percent)
    predicted_values = []
    is_new_death = False
    for a, b in zip(test_result[0], test_result[1]):
        for _a, _b in zip(a*200000, b*200000):
            __a = float(round(_a, 2))
            __b = float(round(_b, 2))
            # To get reasonable results, we need
            # the minimum of the predicted deaths
            # and the average of the predicted cases.
            if is_new_death:
                predicted_values.append(min(__a, __b))
                is_new_death = False
            else:
                predicted_values.append((__a+__b)/2)
                is_new_death = True
    return np.reshape(predicted_values, (-1, 2))


def __train(data: list, activations: list, hidden_layers: list, bias: int, train_percent: float, acceptable_error_margin=0.0, max_epoch=20) -> tuple:
    print('Training Neural Network...')
    train_data = __get_train_data(data, train_percent)
    train_x = (np.array([data[i:i + HIST] for i in range(len(train_data)-HIST)]))/200000
    train_y = (np.array([data[i+HIST] for i in range(len(train_data)-HIST)]))/200000
    train_x2 = train_x.reshape((train_x.shape[0], train_x.shape[1]*train_x.shape[2]))
    neural_net = [len(train_x2[0])+bias]
    for i in hidden_layers:
        neural_net.append(i)
    neural_net.append(len(train_y[0]))
    weight_list = [np.random.random((neural_net[layer+1], neural_net[layer]))*0.5-0.25 for layer in range(len(neural_net)-1)]

    epoch = 0
    error_sum = len(train_x2)
    start_time = time.time()
    while __check_error_margin(acceptable_error_margin, error_sum / len(train_x2)) and epoch<max_epoch:
        error_sum = 0.0
        epoch += 1
        time0 = time.time()
        for inp, out in zip(train_x2, train_y):
            neuron_layer = [list(inp) + [1.0] * bias]
            for l in range(len(neural_net) - 1):
                activation = activations[l]
                neuron_layer.append([activation.fn(sum([neuron_layer[l][i] * weight_list[l][j][i] for i in range(neural_net[l])])) for j in range(neural_net[l + 1])])

            error = [out[j] - neuron_layer[-1][j] for j in range(neural_net[-1])]
            delta = [None for _ in range(len(neural_net) - 1)]
            for l in reversed(range(len(neural_net) - 1)):
                activation = activations[l]
                if l == len(neural_net) - 2:
                    delta[l] = [error[j] * activation.dfn(neuron_layer[-1][j]) for j in range(neural_net[-1])]
                else:
                    delta[l] = [sum([delta[l + 1][j] * weight_list[l + 1][j][i] for j in range(neural_net[l + 2])]) * activation.dfn(neuron_layer[l + 1][i])
                                for i in range(neural_net[l + 1])]

                for i in range(neural_net[l]):
                    for j in range(neural_net[l + 1]):
                        weight_list[l][j][i] += 0.01 * delta[l][j] * neuron_layer[l][i]

            error_sum += sum([error[j] ** 2 for j in range(neural_net[-1])])
        print(f'Epoch {epoch}/{max_epoch} - error: {round(error_sum/len(train_x2), 3)} - elapsed time: {round(time.time()-time0, 3)}s')
    print(f'Total elapsed time: {round(time.time()-start_time, 3)}s')
    return neural_net, weight_list, round(error_sum/len(train_x2), 3)


def __test(data: list, activations: list, neural_net: list, weight_list: list, bias: int, test_percent: float) -> tuple:
    print('Testing Neural Network...')
    test_data = __get_test_data(data, test_percent)
    test_x = (np.array([test_data[i:i+HIST] for i in range(len(test_data)-HIST)]))/200000
    test_y = (np.array([test_data[i+HIST] for i in range(len(test_data)-HIST)]))/200000
    test_x2 = test_x.reshape((test_x.shape[0], test_x.shape[1] * test_x.shape[2]))

    predicted = []
    for inp, out in zip(test_x2, test_y):
        neuron_layer = [list(inp) + [1.0] * bias]
        for l in range(len(neural_net) - 1):
            activation = activations[l]
            neuron_layer.append([activation.fn(sum([neuron_layer[l][i] * weight_list[l][j][i] for i in range(neural_net[l])])) for j in range(neural_net[l + 1])])

        predicted.append(neuron_layer[-1])
    predicted = np.array(predicted)
    print('Prediction MSE: ', mean_squared_error(test_y, predicted))
    print('Stupid MSE: ', mean_squared_error(test_y[1:], test_y[:-1]))
    return test_y, predicted


def __check_error_margin(margin: float, error: float):
    if margin != 0:
        return error > margin
    else:
        return True


def __save(train_result: tuple):
    if not exists('export'):
        mkdir('export')

    filename = join('export', f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')} {str(train_result[2])}.json")
    print(f'Saving as {filename}...')
    weight_list = []
    for i in range(len(train_result[1])):
        weight_list.append(train_result[1][i].tolist())
    data = {
        'neural_net': train_result[0],
        'weight_list': weight_list
    }
    json_string = json.dumps(data, indent='  ')
    with open(filename, 'w') as export:
        export.write(json_string)
    print('Successfully saved the network!')


def __load(filename: str) -> tuple:
    print(f'Loading {filename}...')
    with open(filename) as f:
        jsonfile = json.load(f)
        neural_net = jsonfile['neural_net']
        loaded_weight_list = jsonfile['weight_list']
        weight_list = []
        for i in range(len(loaded_weight_list)):
            weight_list.append(np.array(loaded_weight_list[i]))
        return neural_net, weight_list
