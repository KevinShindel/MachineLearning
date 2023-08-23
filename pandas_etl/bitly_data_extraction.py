from urllib.request import urlopen

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns


def main():
    online_file_path = 'https://raw.githubusercontent.com/wesm/pydata-book/3rd-edition/datasets/bitly_usagov/example.txt'
    offline_file_path = '../../MachineLearning/dataset/bitly_example_data.txt'

    # file_columns = ['agent', 'country_code', '', 'TimeZone', '']

    data = [json.loads(line) for line in open(offline_file_path)]
    df = pd.DataFrame(data=data)
    total_time_zones = df['tz'].count()
    uniq_time_zones = df['tz'].dropna().unique()

    df['tz'] = df['tz'].fillna('Missing')
    df[df['tz'] == ''] = 'Unknown'
    tz_counts = df['tz'].value_counts()
    print(total_time_zones, len(uniq_time_zones), tz_counts)
    subset = tz_counts[:10]
    sns.barplot(x=subset.values, y=subset.index)
    plt.show()

    # create os column
    df['browser'].fillna('Unknown', inplace=True)

    grouped = df.groupby(['tz', 'os'])
    by_tz_os = grouped.size().unstack().fillna(0)

    sns.barplot(x=by_tz_os.values, y=by_tz_os.index, facecolor='c', edgecolor='w')
    plt.show()


def yield_line_from_file(file):
    for line in file:
        decoded_line = line.decode('utf-8')
        json_line = json.loads(decoded_line)
        yield json_line


def parse_os(df: pd.DataFrame) -> pd.DataFrame:
    df['os'] = np.where(
        df['browser'].str.contains('Windows'), 'Windows',
        np.where(df['browser'].str.contains('iPhone'), 'IOS',
                 np.where(df['browser'].str.contains('iPad'), 'IOS',
                          np.where(df['browser'].str.contains('Macintosh'), 'IOS',
                                   np.where(df['browser'].str.contains('Linux'), 'Linux',
                                            np.where(df['browser'].str.contains('Android'), 'Android',
                                                     'Unknown'))))))
    return df


def parse_browser(df: pd.DataFrame) -> pd.DataFrame:
    # browser agent
    df['agent'] = np.where(df['browser'].str.contains('Mozilla'), 'Mozilla',
                           np.where(df['browser'].str.contains('Opera'), 'Opera',
                                    np.where(df['browser'].str.contains('Google'), 'Google',
                                             'Missing')))
    return df


def group_by_os_and_tz(bit_ly_df):
    grouped = bit_ly_df.groupby(['os', 'timezone'])
    agg_counts = grouped.size().unstack().fillna(0)
    top10 = agg_counts[:10]
    indexer = top10.sum(1).argsort()
    count_subset = top10.take(indexer).stack()
    count_subset.name = 'total'
    count_subset = count_subset.reset_index()
    top10 = count_subset[:10]
    sns.barplot(x='total', y='timezone', hue='os', data=top10)
    plt.show()

def read_online_file():
    online_file_path = 'https://raw.githubusercontent.com/wesm/pydata-book/3rd-edition/datasets/bitly_usagov/example.txt'
    request = urlopen(url=online_file_path)
    parsed_data = yield_line_from_file(request)
    new_column_names = ['browser', 'country_code', 'timezone']
    bit_ly_df = pd.DataFrame(parsed_data)
    bit_ly_df = bit_ly_df[['a', 'c', 'tz']]
    bit_ly_df.columns = new_column_names
    # fill empty values
    bit_ly_df = bit_ly_df.fillna('Unknown')
    bit_ly_df = bit_ly_df.replace('', 'Unknown')

    bit_ly_df = parse_os(bit_ly_df)
    bit_ly_df = parse_browser(bit_ly_df)

    count_tz(bit_ly_df)

    group_by_os_and_tz(bit_ly_df)

    print(bit_ly_df)

def count_tz(df: pd.DataFrame):
    tz_counts = df['timezone'].value_counts()
    print(tz_counts)

    subset = tz_counts[:10]
    sns.barplot(x=subset.values, y=subset.index)
    plt.show()

if __name__ == '__main__':
    read_online_file()
