"""
Description: Investigate Bitly data.
Author: Kevin Shindel
Date: 2024-08-05
"""
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import zipfile as zf


def get_counts(sequence):
    counts = {}
    for x in sequence:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts


def get_counts2(sequence):
    counts = defaultdict(int)
    for x in sequence:
        counts[x] += 1
    counts = sorted(counts.items(), key=lambda i: i[1], reverse=True)
    return counts


def top_counts(count_dict):
    value_key_pairs = [(count, tz) for tz, count in count_dict.items()]
    value_key_pairs.sort(reverse=True)
    return value_key_pairs


def counter_counts(sequence, n=10):
    from collections import Counter
    counts = Counter(sequence)
    return counts.most_common(n)


def main():
    # provide path to file
    # offline_file_path = '../../MachineLearning/dataset/bitly_example_data.txt'
    offline_file_path = '../../dataset/bitly_example_data.zip'

    # open zip file
    z = zf.ZipFile(offline_file_path)
    content = z.open(next(iter(z.namelist())))

    # load data from file and parse it to json
    data = [json.loads(line) for line in content]
    # get all time zones from json data
    time_zones = [rec['tz'] for rec in data if 'tz' in rec]
    print(time_zones[:10])
    # get top 10 time zones by building dictionary
    counts = get_counts(sequence=time_zones)
    print(counts)
    # get top 10 time zones by defaultdict method
    counts2 = get_counts2(sequence=time_zones)
    print(counts2)
    # get top 10 time zones by sorted method
    counts3 = top_counts(count_dict=counts)
    print(counts3)
    # get top 10 time zones by Counter method
    counts4 = counter_counts(sequence=time_zones, n=10)
    print(counts4)

    # create pandas dataframe from json data
    bitly_df = pd.DataFrame(data=data)
    print(bitly_df.info())

    # get all time zones from pandas dataframe
    total_time_zones = bitly_df['tz'].count()
    uniq_time_zones = bitly_df['tz'].value_counts()
    print(total_time_zones, len(uniq_time_zones), uniq_time_zones[:10])

    # fill empty values
    bitly_df['tz'] = bitly_df['tz'].fillna('Missing')
    bitly_df[bitly_df['tz'] == ''] = 'Unknown'

    subset = bitly_df['tz'].value_counts()[:10]

    # create barplot
    sns.barplot(x=subset.values, y=subset.index)
    plt.tight_layout()
    plt.show()

    # get all browser agents
    agents = pd.Series([x.split()[0] for x in bitly_df.a.dropna()])
    print(agents[:5])

    # get top 5 browser agents
    top5 = agents.value_counts()[:5]
    print(top5)

    # creat frame with agents
    data_w_agents = bitly_df[bitly_df.a.notnull()]
    # replace agents with Windows and Not Windows
    # data_w_agents['os'] = np.where(data_w_agents['a'].str.contains('Windows'), 'Windows', 'Not Windows')
    data_w_agents = parse_os(data_w_agents)

    # group by time zone and agent
    by_tz_os = data_w_agents.groupby(['tz', 'os'])
    # count values
    agg_counts = by_tz_os.size().unstack().fillna(0)
    print(agg_counts[:10])

    # created top 10 time zones
    indexer = agg_counts.sum(1).argsort()
    print(indexer[:10])

    # get top 10 time zones
    count_subset = agg_counts.take(indexer)[-10:]
    print(count_subset)

    agg_counts = agg_counts.sum(1).nlargest(10)
    print(agg_counts)

    counts_subset = count_subset.stack()
    counts_subset.name = 'total'
    counts_subset = counts_subset.reset_index()

    # create barplot w top 10 time zones
    sns.barplot(x='total', y='tz', hue='os', data=counts_subset)
    plt.tight_layout()
    plt.show()

    # first way - less efficient
    resulted_df = counts_subset.groupby(['tz']).apply(normal_total)

    # second way - more efficient
    # grouped_df = counts_subset.groupby(['tz'])
    # resulted_df['total'] = counts_subset.total / grouped_df.total.transform('sum')

    sns.barplot(x='normed_total', y='tz', hue='os', data=resulted_df)
    plt.tight_layout()
    plt.title('Normalized total time zones')
    plt.show()


def normal_total(group):
    group['normed_total'] = group.total / group.total.sum()
    return group


def yield_line_from_file(file):
    for line in file:
        decoded_line = line.decode('utf-8')
        json_line = json.loads(decoded_line)
        yield json_line


def parse_os(df: pd.DataFrame) -> pd.DataFrame:
    df['os'] = np.where(
        df['a'].str.contains('Windows'), 'Windows',
        np.where(df['a'].str.contains('iPhone'), 'IOS',
                 np.where(df['a'].str.contains('iPad'), 'IOS',
                          np.where(df['a'].str.contains('Macintosh'), 'IOS',
                                   np.where(df['a'].str.contains('Linux'), 'Linux',
                                            np.where(df['a'].str.contains('Android'), 'Android',
                                                     'Unknown'))))))
    return df


def parse_browser(df: pd.DataFrame) -> pd.DataFrame:
    # browser agent
    df['agent'] = np.where(df['browser'].str.contains('Mozilla'), 'Mozilla',
                           np.where(df['browser'].str.contains('Opera'), 'Opera',
                                    np.where(df['browser'].str.contains('Google'), 'Google',
                                             'Missing')))
    return df


if __name__ == '__main__':
    main()
