import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from adtk.detector import ThresholdAD, OutlierDetector
from adtk.visualization import plot
from pandas import DataFrame
from pandas import read_csv
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose


def main():
    price_col = 'buybox_price'

    sns.set(style="darkgrid")
    df = read_csv('../dataset/keepa_data.csv',
                  index_col=['date_at'],
                  parse_dates=True,
                  usecols=['date_at', 'buybox_price'])

    df = df[df.index.year < 2021]

    sns.lineplot(data=df, x=df.index, y=price_col)
    df.hist()
    plt.show()

    # resample to daily
    df = df.resample('D').max()
    df[df[price_col] == 0] = np.nan
    df = df.fillna(method='ffill')

    # get quantile for price
    low_price_threshold = df[price_col].quantile(0.05)
    high_price_threshold = df[price_col].quantile(0.95)

    # replace outliers with quantile
    df[df[price_col] > high_price_threshold] = high_price_threshold
    df[df[price_col] < low_price_threshold] = low_price_threshold

    sns.lineplot(data=df, x=df.index, y=price_col)
    df.hist()
    plt.show()

    threshold_ad = ThresholdAD(high=high_price_threshold, low=low_price_threshold)
    th_anomalies = threshold_ad.detect(df)
    plot(df, anomaly=th_anomalies,
         ts_linewidth=1,
         ts_markersize=3,
         anomaly_markersize=5,
         anomaly_color='red',
         anomaly_tag="marker"
         )
    plt.show()

    outlier_detector = OutlierDetector(LocalOutlierFactor(contamination=0.05))
    od_anomalies = outlier_detector.fit_detect(df)
    od_anomalies = od_anomalies.to_frame(name='Y')

    plot(df, anomaly=od_anomalies,
         ts_linewidth=2,
         ts_markersize=3,
         anomaly_markersize=5,
         anomaly_color='red',
         anomaly_tag='marker',
         ts_alpha=0.3,
         curve_group='all')
    plt.show()

    seasonal(data=df)

    result_m = seasonal_decompose(df[price_col], model='multiplicative', period=12)
    plt.rcParams.update({'figure.figsize': (10, 10)})
    result_m.plot().suptitle('Мультипликативная модель')
    plt.show()

    train_df = df['2017': '2019']  # данные которые можно использовать для тренировки
    print(train_df.head(5))

    test = df['2020': '2020']  # тестовые данные который используются для проверки обучения модели
    print(test.head(5))

    # сезонность 12 периодов методотом Хольта Винтерса
    fit1 = ExponentialSmoothing(endog=train_df, seasonal_periods=12, trend='add', seasonal='mul').fit()
    print(fit1.params)

    forecast1 = fit1.forecast(steps=4)  # предсказать на 4 месяца
    forecast1.plot()
    plt.show()

    # постройка графика на основании осонвного фрейма
    ax = df.plot(figsize=(15, 6), color='black', title='Прогноз методотом Хольта Винтерса',
                 label='фактические значения')

    # показать смоделированые значения
    fit1.fittedvalues.plot(ax=ax, style='--', color='red', label='Смоделированые значения')

    # предсказать 12 периодов ( 1 год )
    fit1.forecast(12).plot(ax=ax, style='--', color='green', label='Предсказаные значения')
    ax.legend(loc='best')
    plt.show()  # чёрные - фактические значения, красные - смоделированые значения, зелёные - предсказаные.

    fit2 = ExponentialSmoothing(endog=train_df, seasonal_periods=12, trend='mul',
                                seasonal='mul').fit()  # сезонность 12 периодов методотом Хольта Винтерса
    fit3 = ExponentialSmoothing(endog=train_df, seasonal_periods=12, trend='mul',
                                seasonal='add').fit()  # сезонность 12 периодов методотом Хольта Винтерса
    fit4 = ExponentialSmoothing(endog=train_df, seasonal_periods=12, trend='add',
                                seasonal='add').fit()  # сезонность 12 периодов методотом Хольта Винтерса

    # постройка графика на основании осонвного фрейма
    ax = df.plot(figsize=(15, 6), color='black', title='Прогноз методотом Хольта Винтерса',
                 label='фактические значения')

    # показать смоделированые значения
    fit1.fittedvalues.plot(ax=ax, style='--', color='red', label='Смоделированые значения')

    # предсказать 12 периодов ( 1 год )
    fit1.forecast(12).plot(ax=ax, style='--', color='green', label='Предсказаные значения')

    # показать смоделированые значения
    fit2.fittedvalues.plot(ax=ax, style='--', color='blue', label='Смоделированые значения')

    # предсказать 12 периодов ( 1 год )
    fit2.forecast(12).plot(ax=ax, style='--', color='purple', label='Предсказаные значения')
    ax.legend(loc='best')

    # чёрные - фактические значения, красные - смоделированые значения, зелёные - предсказаные.
    plt.show()

    # постройка графика на основании осонвного фрейма
    ax = df.plot(figsize=(15, 6), color='black', title='Прогноз методотом Хольта Винтерса',
                 label='фактические значения')

    # показать смоделированые значения
    fit1.fittedvalues.plot(ax=ax, style='--', color='red', label='Смоделированые значения')

    # предсказать 12 периодов ( 1 год )
    fit1.forecast(12).plot(ax=ax, style='--', color='green', label='Предсказаные значения')

    # показать смоделированые значения
    fit3.fittedvalues.plot(ax=ax, style='--', color='blue', label='Смоделированые значения')

    # предсказать 12 периодов ( 1 год )
    fit3.forecast(12).plot(ax=ax, style='--', color='purple', label='Предсказаные значения')

    ax.legend(loc='best')
    plt.show()  # чёрные - фактические значения, красные - смоделированые значения, зелёные - предсказаные.

    ax = df.plot(figsize=(15, 6), color='black', title='Прогноз методотом Хольта Винтерса',
                 label='фактические значения')  # постройка графика на основании осонвного фрейма

    # показать смоделированые значения
    fit1.fittedvalues.plot(ax=ax, style='--', color='red', label='Смоделированые значения')

    # предсказать 12 периодов ( 1 год )
    fit1.forecast(12).plot(ax=ax, style='--', color='green', label='Предсказаные значения')

    # показать смоделированые значения
    fit4.fittedvalues.plot(ax=ax, style='--', color='blue', label='Смоделированые значения')

    # предсказать 12 периодов ( 1 год )
    fit4.forecast(12).plot(ax=ax, style='--', color='purple', label='Предсказаные значения')

    ax.legend(loc='best')
    plt.show()  # чёрные - фактические значения, красные - смоделированые значения, зелёные - предсказаные.


def seasonal(data: DataFrame):
    plt.figure(figsize=(19, 8), dpi=160)
    for i, y in enumerate(data.index.year.unique()):
        plt.plot(list(range(1, len(data[data.index.year == y])+1)),
                 data[data.index.year == y][data.columns[0]].values, label=y)
    plt.title("Сезонность по периодам")
    plt.legend(loc="best")
    plt.show()


if __name__ == '__main__':
    main()
