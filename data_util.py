from datetime import date, datetime
from os import listdir, remove, mkdir
from os.path import exists, isfile, join

import numpy as np
import re
import pandas as pd
import requests as requests

file = join('data', date.today().strftime('%Y-%m-%d') + '.csv')


def get_latest(iso_codes: list[str]) -> tuple[list, str]:
    """
    :param iso_codes: ISO codes of countries we want to train the network on
    :return: tuple of (latest covid data of summed daily new cases and new deaths, earliest occurrence of covid) for the listed countries
    """
    __delete_old()
    __download(iso_codes)
    csv_file = open(file)
    lines = csv_file.readlines()
    csv_file.close()
    data = [line.strip().split(sep=',') for line in lines]
    data = sorted(data, key=lambda x: datetime.strptime(x[3], '%Y-%m-%d'))
    iso_codes_length = len(iso_codes)
    earliest_occurrence = __find_earliest_occurrence(data, iso_codes_length)
    indexes = [5, 8]  # new_cases, new_deaths
    data = [d for d in data if d[3] >= earliest_occurrence]  # first occurrence of covid in all countries we're testing
    data = [[float(line[i]) if __is_number(line[i]) else 0 for i in indexes] for line in data]
    output = []
    for i in range(int(len(data) / iso_codes_length)):
        j = i * iso_codes_length
        d = np.array(data[j:j + iso_codes_length])
        output.append(d.sum(axis=0))
    return output, earliest_occurrence


def __find_earliest_occurrence(data: list, iso_codes_length: int) -> str:
    dates = []
    for line in data:
        dates.append(line[3])
        if dates.count(line[3]) == iso_codes_length:
            return line[3]
    return '2021-01-01'


def get_7day_avg(data: list) -> list:
    """
    :param data: latest covid data of summed daily new cases and new deaths
    :return: 7 day rolling average of the data
    """
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
    if not exists('data'):
        mkdir('data')
    files = [f for f in listdir('data') if isfile(join('data', f))]
    for f in files:
        if f.endswith('.csv'):
            filedate = date_regex.match(f).group()
            if (datetime.today() - datetime.strptime(filedate, '%Y-%m-%d')).days > 7:
                remove(join('data', f))


def __download(iso_codes: list[str]):
    if not exists(file):
        print("Downloading today's latest data...")
        request = requests.get('https://covid.ourworldindata.org/data/owid-covid-data.csv')
        lines = request.text.split('\n')

        csv_file = open(file, 'x')
        for line in lines:
            iso = line.split(',')[0]
            if any(iso == code for code in iso_codes):
                csv_file.write(line + '\n')
        csv_file.close()
    else:
        csv_file = open(file)
        lines = csv_file.readlines()
        csv_file.close()
        if lines[-1].split(',')[3] != datetime.strftime(datetime.today(), '%Y-%m-%d'):
            print("Data update required...")
            remove(file)
            __download(iso_codes)
