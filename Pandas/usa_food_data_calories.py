import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    """ Investigate USA food data """
    path = '../dataset/usa_food.zip'
    food_df = pd.read_json(path, compression='zip')

    # show first 1 row
    first_item = food_df.iloc[0]
    print(first_item)

    # load only columns we need
    info_food_df = food_df[['description', 'group', 'id', 'manufacturer']]
    print(food_df.head())

    # count food groups
    top10_total_food_groups = pd.value_counts(info_food_df.group)[:10]
    print(top10_total_food_groups)
    # plot top 10 food groups
    sns.set(style="darkgrid")
    sns.barplot(x=top10_total_food_groups.index, y=top10_total_food_groups.values, alpha=0.9)
    plt.title('Frequency Distribution of Top 10 Food Groups')
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Food Group', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # explode nutrition data
    nutrients_dict_df = food_df[['id', 'nutrients']].explode('nutrients').reset_index(drop=True)

    nutrients_normalized_df = pd.json_normalize(nutrients_dict_df.pop('nutrients')).reset_index(drop=True)

    nutrients_df = pd.concat([nutrients_dict_df, nutrients_normalized_df], axis=1). \
        rename(columns={'description': 'nutrient', 'group': 'nutgroup'})

    # merge with food_df
    df = pd.merge(food_df[['id', 'description', 'manufacturer', 'group']], nutrients_df, on='id', how='outer'). \
        drop_duplicates(subset=['id', 'nutrient', 'description', 'group', 'nutgroup'])
    print(df.head())
    result = df.groupby(['nutrient', 'nutgroup'])['value'].quantile(0.5)

    result['Zinc, Zn'].sort_values().plot(kind='barh')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
