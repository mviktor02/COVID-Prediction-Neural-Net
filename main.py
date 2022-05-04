import matplotlib.pyplot as plt
import numpy as np

from activation import sigmoid
from data_util import get_latest, get_7day_avg
from neural_net import apply_simple

data = get_7day_avg(get_latest())
predicted = apply_simple(data, [100, 50], sigmoid(), load_file='export/2022-05-04 14-43-06 0.006.json')
plot_data = np.concatenate((data, predicted), axis=0)

lines = plt.plot(plot_data)
plt.legend(lines, ['New Cases', 'New Deaths'])

plt.xlabel('Days since 2020-03-06')
plt.ylabel('7 day average')
plt.show()
