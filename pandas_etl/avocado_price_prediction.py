import pandas as pd
from matplotlib import pyplot as plt


def main():
    # avocado price prediction
    avocado_df = pd.read_csv('../dataset/avocado.csv',
                             parse_dates=True,
                             index_col='Date')
    avocado_df = avocado_df.drop(avocado_df.columns[0], axis=1)

    albany_avocado_df = avocado_df[avocado_df['region'] == 'Albany']
    albany_avocado_df = albany_avocado_df.drop('region', axis=1)

    albany_avocado_df: pd.DataFrame = (albany_avocado_df.
                                       sort_index().
                                       reset_index().
                                       drop_duplicates(subset='Date', keep='last').
                                       set_index('Date').
                                       resample('D').
                                       ffill())

    print(albany_avocado_df.describe())
    print(albany_avocado_df.info())

    albany_avocado_df['AveragePrice'].plot()
    plt.show()




if __name__ == '__main__':
    main()
