from datetime import date, datetime, timedelta
from os import listdir, remove
from os.path import exists, isfile, join
import numpy as np
import re

import pandas as pd
import requests as requests

file = join('data', date.today().strftime('%Y-%m-%d') + '.csv')
valid_iso_codes = ['HUN', 'AUT', 'SVK', 'UKR', 'ROU', 'SRB', 'HRV', 'SVN']


def get_latest() -> list:
    __delete_old()
    __download()
    csv_file = open(file)
    lines = csv_file.readlines()
    csv_file.close()
    data = [line.strip().split(sep=',') for line in lines]
    data = sorted(data, key=lambda x: datetime.strptime(x[3], '%Y-%m-%d'))
    indexes = [5, 8]  # new_cases, new_deaths
    data = [d for d in data if d[3] >= '2020-03-06']  # first occurrence of covid in all countries we're testing
    data = [[float(line[i]) if __is_number(line[i]) else 0 for i in indexes] for line in data]
    to_sum = len(valid_iso_codes)
    output = []
    for i in range(int(len(data)/to_sum)):
        j = i*to_sum
        d = np.array(data[j:j+to_sum])
        output.append(d.sum(axis=0))
    return output


def get_7day_avg(data: list) -> list:
    avg = pd.DataFrame(np.array(data)).rolling(7, axis=0, min_periods=1).mean().values
    return [[float(a[0]), float(a[1])] for a in avg]


def __is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def __delete_old():
    date_regex = re.compile(r'\d\d\d\d-\d\d-\d\d')
    files = [f for f in listdir('data') if isfile(join('data', f))]
    for f in files:
        if f.endswith('.csv'):
            filedate = date_regex.match(f).group()
            if datetime.strptime(filedate, '%Y-%m-%d') + timedelta(days=7) <= datetime.today():
                remove(f)


def __download():
    if not exists(file):
        request = requests.get('https://covid.ourworldindata.org/data/owid-covid-data.csv')
        lines = request.text.split('\n')

        csv_file = open(file, 'x')
        for line in lines:
            iso = line.split(',')[0]
            if any(iso == code for code in valid_iso_codes):
                csv_file.write(line + '\n')
        csv_file.close()
