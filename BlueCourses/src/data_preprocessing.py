"""
Description: This file contains the functions that are used to preprocess the data.
For documentation see 2_Data_Preprocessing.md
Author: Kevin Shindel
Date: 2024-01-05
"""
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # Determine type of variables

    # Example of Continuous variables (DataFrame with 7 columns and 1000 rows)
    # First column contains the ID of the person [0, 1000]
    # Second column contains the age of the person [18, 65]
    # Third column contains the income of the person [10000, 100000]
    # Fourth column contains the martial status [Married, Single, Divorced]
    # Fifth column contains sex of the person [Male, Female]
    # Six column contains occupation [Student, Employee, Self Employed]
    # Seven columns determine is person is good customer or not [0, 1]
    df = pd.DataFrame({
        'ID': np.arange(1000),
        'Age': np.random.randint(18, 65, 1000),
        'Income': np.random.randint(10000, 100000, 1000),
        'Marital Status': np.random.choice(['Married', 'Single', 'Divorced'], size=1000),
        'Sex': np.random.choice(['Male', 'Female'], size=1000),
        'Occupation': np.random.choice(['Student', 'Employee', 'Self Employed'], size=1000),
        'Good Customer': np.random.randint(0, 2, 1000)
    })

    # Categorize the variables
    df['Marital Status'] = df['Marital Status'].astype('category')
    df['Occupation'] = df['Occupation'].astype('category')
    df['Sex'] = df['Sex'].astype('category')


    # Make df Over-sampled
    over_sampled_df = pd.concat([df]*10, ignore_index=True)

    # make df Under-sampled
    under_sampled_df = df.sample(frac=0.1)

    # Make df with missing values
    df_with_missing_values = df.copy()
    df_with_missing_values.loc[0:100, 'Age'] = np.nan
    df_with_missing_values.loc[200:300, 'Income'] = np.nan
    df_with_missing_values.loc[400:500, 'Marital Status'] = np.nan
    df_with_missing_values.loc[600:1000, 'Occupation'] = np.nan
    df_with_missing_values.loc[800:900, 'Sex'] = np.nan

    # Make df with outliers
    df_with_outliers = df.copy()
    df_with_outliers.loc[0:100, 'Age'] = 100
    df_with_outliers.loc[200:300, 'Income'] = 1000000

    # Make df with duplicates
    df_with_duplicates = pd.concat([df]*2, ignore_index=True)

    # Make df with irrelevant columns
    df_with_irrelevant_columns = df.copy()
    df_with_irrelevant_columns['Irrelevant Column'] = np.random.randint(0, 2, 1000)


    # count number of good and bad customers
    good = df['Good Customer'].value_counts()[1]
    bad = df['Good Customer'].value_counts()[0]

    # determine optimal odds for under and over sampling
    odds = good/bad

    # Undersampled
    under_sampled_df = df[df['Good Customer'] == 1].sample(n=bad, replace=True)

    # Oversampled
    over_sampled_df = df[df['Good Customer'] == 0].sample(n=good, replace=True)

    # combine under and over sampled dataframes
    combined_df = pd.concat([under_sampled_df, over_sampled_df], ignore_index=True, axis=0)

    # Example of SMOTE technique
    smote_coeff = 0.5
    smote_df = df.copy()

    # determine number of samples to generate
    n_samples = int(smote_coeff * bad)

    # generate synthetic samples
    synthetic_samples = pd.DataFrame()
    for i in range(n_samples):
        sample = df[df['Good Customer'] == 1].sample(n=1, replace=True)
        sample['Good Customer'] = 0
        synthetic_samples = pd.concat([synthetic_samples, sample], ignore_index=True)

    # combine original and synthetic samples
    smote_df = pd.concat([df, synthetic_samples], ignore_index=True, axis=0)

    # Data visualization
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Plot the distribution of the age
    sns.histplot(df['Age'])
    plt.show()

    # Plot the distribution of the income
    sns.histplot(df['Income'])
    plt.show()

    # Plot the distribution of the marital status
    sns.countplot(x='Marital Status', data=df)
    plt.show()

    # Deal with Descriptive Statistics
    # Get the descriptive statistics of the age
    print(df['Age'].describe())

    # Find quantiles of the income
    print(df['Income'].quantile([0.25, 0.5, 0.75]))

    # Find mean of the income
    print(df['Income'].mean())

    # find median of the income
    print(df['Income'].median())

    # Find mode of the income
    print(df['Income'].mode())

    # Find variance of the income
    print(df['Income'].var())

    # Missing Values techniques


