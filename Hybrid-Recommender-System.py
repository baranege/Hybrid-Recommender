# Hybrid Recommender
# User based and Item Based Recommender

import pandas as pd
pd.set_option('display.max_columns', 20)

# Task 1: Data Processing

# reading datasets
movie = pd.read_csv("movielens-20m-dataset/movie.csv")
rating = pd.read_csv("movielens-20m-dataset/rating.csv")
df_ = movie.merge(rating, how="left", on="movieId")
df = df_.copy()
df.head()

df.shape

# number of unique titles in ratings
comment_counts = pd.DataFrame(df["title"].value_counts())

# movies rarely rated
rare_movies = comment_counts[comment_counts["title"] <= 1000].index

# exclusion of movies rarely rated
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape
# check number of common movies
common_movies["title"].nunique()

# creating pivot table consisting of so called common movies
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
user_movie_df.shape
user_movie_df.head(10)

# movies are in columns checked
user_movie_df.columns

# Task 2: Determinig the movies that the random picked user watched

# picking up a random user for user based recommendation
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)

#selecting the movies the the random picked user watched
random_user_df = user_movie_df[user_movie_df.index == random_user]
#moving them to a list
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
#verification
#user_movie_df.loc[user_movie_df.index == random_user, user_movie_df.columns == "Ace Ventura: Pet Detective (1994)"]
len(movies_watched)

# Task 3: Finding the other users who watched the same movies

# selecting the movies that random user watched which also includes other users
movies_watched_df = user_movie_df[movies_watched]

# number of movies watched by users to find the similar pattern with random user
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()

#number of movies watched by users
user_movie_count.columns = ["userId", "movie_count"]

# excluding the user who watched less than 20 movies to get similar pattern with random user
# user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False)
# users who watched same amount of movies with random user
# user_movie_count[user_movie_count["movie_count"] == 33].count() # nur 17

# selecting the users who watched more than %60 of movies the the random user watched to get better results
perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

# Task 4: Determining the similar users to random user for recommendation

# creating dataframe consisting of movies watched by random user and other users who watched them
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies.index)],
                      random_user_df[movies_watched]])
final_df.shape

# finding correlations between users
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

# selecting users at least %65 correlated with random user
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)
top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

# rating scores of similar users with random user
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

# Task 5: Calculation of weighted average recommendation score and recommend first 5 movies

# considering rating and correlation together: weighted average
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
# getting the movie IDs and weighted ratings
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()

# 5 movies to recommend (user-based)
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 4].sort_values("weighted_rating", ascending=False)
movies_to_be_recommend = movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"]
movies_to_be_recommend.head(5)

# Task 6: Item based recommendation based on recently watched and highly graded movie by random picked user

# getting the movie ID of recently watched and highly graded movie by random picked user
movie_id = rating[(rating["userId"] == random_user) & (rating["rating"] ==  5.0)].\
sort_values(by = "timestamp", ascending = False)["movieId"][0:6].values[0]

# 5 movies to recommend (item-based)
movie_name = movie[movie["movieId"]== movie_id]["title"]
movie_name = user_movie_df[movie_name]
movies_from_item_based = user_movie_df.corrwith(movie_name).sort_values(ascending=False)
movies_from_item_based[1:6].index
