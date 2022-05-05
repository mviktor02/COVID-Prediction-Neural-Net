from datetime import datetime
from activation import sigmoid
from data_util import get_latest, get_7day_avg
from neural_net import apply_simple

import matplotlib.pyplot as plt
import numpy as np

latest = get_latest(['HUN', 'AUT', 'SVK', 'UKR', 'ROU', 'SRB', 'HRV', 'SVN'])
earliest_occurrence = latest[1]
data = get_7day_avg(latest[0])
predicted = apply_simple(data, [100, 50], sigmoid())
plot_data = np.concatenate((data, predicted), axis=0)

lines = plt.plot(plot_data, label=['New Cases', 'New Deaths'])
today = (datetime.today()-datetime.strptime(earliest_occurrence, '%Y-%m-%d')).days
plt.axvline(x=today, color='r', label='Today')
plt.legend()

plt.xlabel(f'Days since {earliest_occurrence}')
plt.ylabel('7 day average')
plt.show()
