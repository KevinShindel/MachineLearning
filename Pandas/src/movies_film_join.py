"""
Description: Investigate movie data.
Join dataframes and calculate mean rating.
Author: Kevin Shindel
Date: 2024-08-05
"""
import pandas as pd

users_path = '../../dataset/grouplens_1m_users.zip'
ratings_path = '../../dataset/grouplens_1m_ratings.zip'
movies_path = '../../dataset/grouplens_1m_movies.zip'


def main():
    users_headers = ['user_id', 'gender', 'age', 'occupation', 'zip']
    users_raw_df = pd.read_csv(users_path, sep='::', engine='python', header=None, names=users_headers,
                               encoding='latin-1', dtype={'zip': str}, na_values='?')
    rating_headers = ['user_id', 'movie_id', 'rating', 'timestamp']

    ratings_raw_df = pd.read_csv(ratings_path, sep='::', engine='python', header=None, names=rating_headers,
                                 encoding='latin-1', na_values='?')
    movies_headers = ['movie_id', 'title', 'genres']
    movies_raw_df = pd.read_csv(movies_path, sep='::', engine='python', header=None, names=movies_headers,
                                encoding='latin-1', na_values='?')

    # Joining the dataframes
    combined_df = pd.merge(
        pd.merge(ratings_raw_df, users_raw_df),
        movies_raw_df)

    first_row = combined_df.iloc[0]
    print(first_row)

    # calc mean rating for each movie by gender
    mean_ratings = combined_df.pivot_table(values='rating', index='title', columns='gender', aggfunc='mean')

    # filter movies with more than 250 ratings
    # this records equals count_values(subset='title')
    ratings_by_title = combined_df.groupby('title').size().sort_values(ascending=False)
    active_ratings = ratings_by_title.index[ratings_by_title >= 250]

    # select rows on the index
    mean_ratings = mean_ratings.loc[active_ratings]

    # calc rating difference
    mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']

    # sort by diff
    sorted_by_diff = mean_ratings.sort_values(by='diff')

    top_10_male_disliked = sorted_by_diff[:10]
    top_10_female_disliked = sorted_by_diff[::-1][:10]
    print(top_10_male_disliked, top_10_female_disliked)

    # calc std deviation of rating grouped by title
    rating_std_by_title = combined_df.groupby('title')['rating'].std()

    # active title
    rating_std_by_title = ratings_by_title.loc[active_ratings]

    rating_std_by_title = rating_std_by_title.sort_values(ascending=False)[:10]


if __name__ == '__main__':
    main()
