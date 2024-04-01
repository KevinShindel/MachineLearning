import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile as zf


def yield_data(files):
    columns = ['name', 'sex', 'births']

    for file in files:
        frame = pd.read_csv(file, names=columns)
        frame['year'] = file.name[-8:-4]
        frame['year'] = frame['year'].astype(int)
        yield frame


def add_prop(group):
    group['prop'] = group.births / group.births.sum()
    return group


def get_top_1000(group):
    return group.sort_values(by='births', ascending=False)[:1000]


def main():
    # load data from csv
    file_name = '../dataset/usa_names.zip'
    z = zf.ZipFile(file_name)
    files = [z.open(file) for file in z.namelist()]

    data = yield_data(files)
    df = pd.concat(data, ignore_index=True)

    print(df.head())

    total_births = df.pivot_table(values='births', index='year', columns='sex', aggfunc=sum)

    print(total_births.tail())

    sns.set()
    total_births.plot()
    plt.ylabel('total births per year')
    plt.xlabel('year')
    plt.show()

    # add prop column
    df = df.groupby(['year', 'sex']).apply(add_prop).reset_index(drop=True)

    # sanity check
    sanity_check_df = df.groupby(['sex', 'year']).prop.sum()
    print(sanity_check_df)

    top1000 = df.groupby(['sex', 'year']).apply(get_top_1000).reset_index(drop=True)

    print(top1000.head())

    # tendencies in names
    boys = top1000[top1000.sex == 'M']
    # girls = top1000[top1000.sex == 'F']

    total_births = top1000.pivot_table(values='births', index='year', columns='name', aggfunc=sum)
    print(total_births.head())
    print(total_births.info())

    # plot some names
    subset = total_births[['John', 'Harry', 'Mary', 'Marilyn']]
    subset.plot(subplots=True, figsize=(12, 10), grid=False, title='Number of births per year')
    plt.show()

    # measure increase in naming diversity
    table = top1000.pivot_table(values='prop', index='year', columns='sex', aggfunc=sum)

    max_ticks = top1000.year.max()
    min_ticks = top1000.year.min()
    step = 25

    # part of births represented by top 1000 names
    table.plot(title='Sum of table1000.prop by year and sex',
               yticks=np.linspace(0, 1.2, 13),
               xticks=range(min_ticks, max_ticks, step),
               figsize=(12, 10))
    plt.show()

    # male children births in 2010
    boys_2010 = boys[boys.year == 2010]

    prop_cum_sum = boys_2010.sort_values(by='prop', ascending=False).prop.cumsum()

    top10_in_2010 = prop_cum_sum.searchsorted(0.5) + 1
    top10_in_1900 = boys[boys.year == 1900].sort_values(by='prop', ascending=False).prop.cumsum().searchsorted(0.5) + 1

    print(top10_in_2010, top10_in_1900)

    # calculate diversity of names in top 50% of births
    diversity = top1000.groupby(['year', 'sex']).apply(get_quantile_count)

    #
    resulted_set = diversity.unstack('sex')
    resulted_set.plot(title='Number of popular names in top 50%')
    plt.show()

    # last letter revolution
    last_letters = df.name.map(lambda x: x[-1])
    last_letters.name = 'last_letter'

    # pivot table
    table = df.pivot_table(values='births', index=last_letters,
                           columns=['sex', 'year'], aggfunc=sum)

    sub_table = table.reindex(columns=[1910, 1960, 2010], level='year')
    sub_table.plot(subplots=True, figsize=(12, 10), grid=False, title='Number of births per year')
    plt.show()

    letter_prop = sub_table / sub_table.sum()

    print(letter_prop.head())

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    letter_prop['F'].plot(kind='bar', rot=0, ax=axes[0], title='female')
    letter_prop['M'].plot(kind='bar', rot=0, ax=axes[1], title='male', legend=False)
    plt.show()

    letter_prop = table / table.sum()
    dny_ts = letter_prop.loc[['d', 'n', 'y'], 'M'].T
    print(dny_ts.head())

    dny_ts.plot()
    plt.show()

    # man names that converted to female names
    all_names = pd.Series(top1000.name.unique())
    lesley_like = all_names[all_names.str.lower().str.contains('lesl')]
    print(lesley_like)

    filtered = top1000[top1000.name.isin(lesley_like)]
    print(filtered.groupby('name').births.sum())

    table = filtered.pivot_table(values='births', index='year', columns='sex', aggfunc=sum)
    table = table.div(table.sum(1), axis=0)
    print(table.tail())

    table.plot(style={'M': 'k-', 'F': 'k--'})
    plt.show()


def get_quantile_count(group, q=0.5):
    group = group.sort_values(by='prop', ascending=False)
    return group.prop.cumsum().searchsorted(q) + 1


if __name__ == '__main__':
    main()
