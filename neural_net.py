import time

import numpy as np
from sklearn.metrics import mean_squared_error

from activation import Activation, sigmoid

HIST = 7


def __get_train_data(data: list):
    sixtypercent = int(len(data)*0.6)
    return data[:sixtypercent]


def __get_test_data(data: list):
    sixtypercent = int(len(data)*0.6)
    return data[sixtypercent:]


def apply_simple(data: list, hidden_layers: list, activation: Activation, acceptable_error_rate=0.0001, save=False, bias=1) -> list:
    activations = [activation for _ in range(len(hidden_layers)+3)]
    return apply(data, hidden_layers, activations, acceptable_error_rate, save, bias)


def apply(data: list, hidden_layers: list, activations: list, acceptable_error_rate=0.0001, save=False, bias=1) -> list:
    train_result = __train(data, activations, hidden_layers, bias, acceptable_error_rate)
    if save:
        __save(train_result)

    test_result = __test(data, activations, train_result[0], train_result[1], bias)
    print('Predicted values:')
    print('New Cases; New Deaths')
    predicted = []
    for a, b in zip(test_result[0], test_result[1]):
        for _a, _b in zip(a*200000, b*200000):
            new_cases = round(_a, 2)
            new_deaths = round(_b, 2)
            predicted.append([new_cases, new_deaths])
            print(new_cases, new_deaths, end='; ')
        print()
    return predicted


def __train(data: list, activations: list, hidden_layers: list, bias: int, acceptable_error_margin=0.0, max_epoch=20) -> tuple:
    print('Training Neural Network...')
    train_data = __get_train_data(data)
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
    return neural_net, weight_list


def __test(data: list, activations: list, neural_net: list, weight_list: list, bias: int) -> tuple:
    print('Testing Neural Network...')
    test_data = __get_test_data(data)
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
    # TODO save the train result to file
    print('Saving...')
