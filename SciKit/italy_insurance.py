import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

if __name__ == '__main__':
    CURRENT_RATE = 0.15  # Insurance rate for 1 class air transportation

    #  Read the data
    temperature_df = pd.read_csv('../dataset/temperature_anomaly.csv',
                                 index_col='year',
                                 parse_dates=True,
                                 usecols=['year', 'avg_anomaly_temp'])
    MAX_YEAR = temperature_df.index.max().year

    # set insurance rate for current year
    temperature_df['insurance_rate'] = np.nan
    temperature_df.loc[temperature_df.index.year == MAX_YEAR, 'insurance_rate'] = CURRENT_RATE

    # normalize temperature data from min year to max year
    min_val = temperature_df['avg_anomaly_temp'].min()
    max_val = temperature_df['avg_anomaly_temp'].max()

    temperature_df['avg_anomaly_temp'] = (temperature_df['avg_anomaly_temp'] - min_val) / (max_val -
                                                                                   min_val)

    # calculate linear relation for insurance rate
    temperature_df['anomaly_coeff'] = np.nan
    temperature_df.loc[temperature_df.index.year == MAX_YEAR, 'anomaly_coeff'] = temperature_df.loc[
        temperature_df.index.year == MAX_YEAR, 'avg_anomaly_temp'] / CURRENT_RATE

    # get anomaly coefficient
    ANOMALLY_COEFF = temperature_df.loc[temperature_df.index.year == MAX_YEAR, 'anomaly_coeff'].values[0]

    # calculate insurance rate for other years
    temperature_df['insurance_rate'] = temperature_df['avg_anomaly_temp'] / ANOMALLY_COEFF

    # plot emissions and insurance rate and group by 25 years
    plt.plot(temperature_df.index.year, temperature_df['avg_anomaly_temp'], label='Temperature Anomaly')
    plt.plot(temperature_df.index.year, temperature_df['insurance_rate'], label='Insurance Rate')
    plt.legend()
    plt.show()

    # Read C02 data
    c02_df = pd.read_csv('../dataset/co2_emission.csv',
                          index_col='year',
                          parse_dates=True)

    MAX_YEAR = c02_df.index.max().year

    # set insurance rate for current year
    c02_df['insurance_rate'] = np.nan
    c02_df.loc[c02_df.index.year == MAX_YEAR, 'insurance_rate'] = CURRENT_RATE

    # calculate linear relation for insurance rate
    c02_df['anomaly_coeff'] = np.nan
    c02_df.loc[c02_df.index.year == MAX_YEAR, 'anomaly_coeff'] = c02_df.loc[
        c02_df.index.year == MAX_YEAR, 'emissions'] / CURRENT_RATE

    # get anomaly coefficient
    ANOMALLY_COEFF = c02_df.loc[c02_df.index.year == MAX_YEAR, 'anomaly_coeff'].values[0]

    # calculate insurance rate for other years
    c02_df['insurance_rate'] = c02_df['emissions'] / ANOMALLY_COEFF

    # plot  emissions and insurance rate and group by 25 years
    plt.plot(c02_df.index.year, c02_df['emissions'], label='Emissions')
    plt.plot(c02_df.index.year, c02_df['insurance_rate'], label='Insurance Rate')
    plt.legend()

    plt.xlabel('Emissions')
    plt.ylabel('Insurance Rate')
    plt.title('Relation between emissions and insurance rate')
    plt.show()

    # TODO: Create prediction model to predict CO2 emissions and insurance rate

    # TODO: Create prediction model to predict temperature anomaly and insurance rate

    # disaster insurance

    # Read the data
    disaster_df = pd.read_csv('../dataset/ItalyDisastersDB_1900_2024.csv')

    # fill blanked values with 0 in months and days columns
    disaster_df['Start Month'] = disaster_df['Start Month'].fillna(1)
    disaster_df['Start Day'] = disaster_df['Start Day'].fillna(1)
    disaster_df['End Month'] = disaster_df['End Month'].fillna(1)
    disaster_df['End Day'] = disaster_df['End Day'].fillna(1)

    # create disaster startdate from year, month and day columns
    start_df = disaster_df[['Start Year', 'Start Month', 'Start Day']].rename({'Start Year': 'year',
                                                                               'Start Month': 'month',
                                                                              'Start Day': 'day'}, axis=1)
    # convert floats to integers
    start_df = start_df.astype(int)
    disaster_df['start_date'] = pd.to_datetime(start_df[['year', 'month', 'day']], format='%Y-%m-%d')

    # do the same to end date
    end_df = disaster_df[['End Year', 'End Month', 'End Day']].rename({'End Year': 'year',
                                                                       'End Month': 'month',
                                                                       'End Day': 'day'}, axis=1)
    end_df = end_df.astype(int)
    disaster_df['end_date'] = pd.to_datetime(end_df[['year', 'month', 'day']], format='%Y-%m-%d')

    # calculate disaster duration
    disaster_df['duration'] = (disaster_df['end_date'] - disaster_df['start_date']).dt.days

    # replace negatives and zeros with 1
    disaster_df.loc[disaster_df['duration'] <= 0, 'duration'] = 1

    # calculate disaster insurance rate
    # disaster_df['insurance_rate'] = np.nan

    # group by year and sum Total Damage, Adjusted ('000 US$) and  create a plot
    grouped = disaster_df.groupby('Start Year').agg({
        "Total Damage ('000 US$)": 'sum',
        "Total Damage, Adjusted ('000 US$)": 'sum',
        'Total Deaths': 'sum',
        'No. Injured': 'sum',
        'duration': 'sum',
    })
    plt.plot(grouped.index, grouped["Total Damage ('000 US$)"], label='Total Damages')
    plt.plot(grouped.index, grouped["Total Damage, Adjusted ('000 US$)"], label='Adjusted Damages')
    plt.ylabel('Damages')
    plt.xlabel('Year')
    plt.title('Total Damages and Adjusted Damages per year')
    plt.legend()

    plt.show()