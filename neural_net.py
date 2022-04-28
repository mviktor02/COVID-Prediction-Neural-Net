import random
import time

from activation import Activation, tanh
import numpy as np
from sklearn.metrics import mean_squared_error

HIST = 7


def __get_train_data(data: list):
    seventypercent = int(len(data)*0.7)
    return data[:seventypercent]


def __get_test_data(data: list):
    seventypercent = int(len(data)*0.7)
    return data[seventypercent:]


def apply(data: list, bias=1) -> list:
    train_result = __train(data, tanh(), [100, 50], bias, 0.0001)
    should_save = input('Save this network? (Y/n)')
    if should_save.lower() == 'y':
        __save(train_result)

    test_result = __test(data, tanh(), train_result[0], train_result[1], bias)
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


def __train(data: list, activation: Activation, hidden_layers: list, bias: int, acceptable_error_margin=0.0, max_epoch=20) -> tuple:
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
    delta = [np.zeros(neural_net[layer+1]) for layer in range(len(neural_net)-1)]

    epoch = 0
    error_sum = len(train_x2)
    start_time = time.time()
    while __check_error_margin(acceptable_error_margin, error_sum / len(train_x2)) and epoch<max_epoch:
        error_sum = 0.0
        epoch += 1
        time0 = time.time()
        for inp, out in random.sample([(a, b) for a, b in zip(train_x2, train_y)], len(train_x2)):
            layer = [np.array(list(inp) + [1.0]*bias)]
            for neuron in range(len(neural_net)-1):
                layer.append(activation.fn(np.dot(weight_list[neuron], neural_net[neuron])))
            error = out - layer[-1]
            for neuron in reversed(range(len(neural_net)-1)):
                if neuron == len(neural_net)-2:
                    delta[neuron][:] = error*activation.dfn(layer[-1])
                else:
                    np.dot(delta[neuron+1], weight_list[neuron+1], out=delta[neuron])
                    delta[neuron] *= activation.dfn(layer[neuron+1])

                weight_list[neuron] += 0.01 * delta[neuron].reshape((neural_net[neuron+1], 1))*layer[neuron].reshape((1, neural_net[neuron]))

            error_sum += sum([error[j]**2 for j in range(neural_net[-1])])
        print(f'Epoch {epoch}/{max_epoch} - error: {round(error_sum/len(train_x2), 3)} - elapsed time: {round(time.time()-time0, 3)}ms')
    print(f'Total elapsed time: {round(time.time()-start_time, 3)}ms')
    return neural_net, weight_list


def __test(data: list, activation: Activation, neural_net: list, weight_list: list, bias: int) -> tuple:
    print('Testing Neural Network...')
    test_data = __get_test_data(data)
    test_x = (np.array([test_data[i:i+HIST] for i in range(len(test_data)-HIST)]))/200000
    test_y = (np.array([test_data[i+HIST] for i in range(len(test_data)-HIST)]))/200000
    test_x2 = test_x.reshape((test_x.shape[0], test_x.shape[1] * test_x.shape[2]))

    predicted = []
    for inp, out in zip(test_x2, test_y):
        neuron_layer = [np.array(list(inp) + [bias])]
        for layer in range(len(neural_net)-1):
            neuron_layer.append(activation.fn(np.dot(weight_list)))
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
