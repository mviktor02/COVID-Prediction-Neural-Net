import matplotlib.pyplot as plt
import numpy as np

from data_util import get_latest, get_7day_avg
from neural_net import apply

data = get_7day_avg(get_latest())
predicted = apply(data, 1)
plot_data = np.concatenate((data, predicted), axis=0)

lines = plt.plot(plot_data)
plt.legend(lines, ['New Cases', 'New Deaths'])

plt.xlabel('Days since 2020-03-06')
plt.ylabel('7 day average')
plt.show()
