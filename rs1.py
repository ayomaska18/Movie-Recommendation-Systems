import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from numpy.linalg import norm
from math import sqrt
import argparse
import pickle
import os

def save_model(predictions, filename):
    with open(filename, 'wb') as f:
        pickle.dump(predictions, f)
    print(f"Model saved to {filename}")

def cal_rmse(test_arr, pred_arr):
    se = 0
    count = 0
    num_users = test_arr.shape[0]
    num_items = test_arr.shape[1]

    for r in range(num_users):
        for c in range(num_items):
            actual = test_arr[r, c]
            if actual > 0:
                diff = (pred_arr[r, c] - actual)
                se += diff * diff
                count += 1

    return sqrt(se / count) if count else 0

## Data Preprocessing ##

# Load the Data
def load_data():
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

    # Split the data into training and test sets
    train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)

    user = pd.read_csv('ml-100k/u.user', sep='|', names=['user_id', 'age', 'occupation', 'zip_code'])
    movie_genres = pd.read_csv('ml-100k/u.genre', sep='|', names=['genre', 'genre_id'])
    genres = list(movie_genres['genre'])

    columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL'] + genres
    items = pd.read_csv('ml-100k/u.item', sep='|', header=None, names=columns, encoding='latin-1')

    movie_df = items[['movie_id','title']]

    train_df = pd.merge(train_ratings, movie_df, how='inner', on='movie_id')

    train_df = train_df.groupby(by=['user_id','title'], as_index=False).agg({"rating":"mean"})

    train_df = train_df.pivot(index='user_id', columns='title', values='rating').fillna(0)

    test_df = pd.merge(test_ratings, movie_df, how='inner', on='movie_id')

    test_df = test_df.groupby(['user_id','title'], as_index=False).agg({'rating':'mean'})

    test_df = test_df.pivot(index='user_id', columns='title', values='rating').fillna(0)

    test_df = test_df.reindex(index=train_df.index, columns=train_df.columns).fillna(0)

    return train_df, test_df, items


def calculate_similarity(train_df):
    ## Collaborative Filtering ##
    train_array = train_df.values 
    num_users   = train_array.shape[0]

    # Compute similarities between users
    similarities = np.zeros((num_users, num_users))

    for i in range(num_users):
        for j in range(num_users):
            if i != j:
                u_vec = train_array[i, :]
                v_vec = train_array[j, :]

                dot = np.dot(u_vec, v_vec)
                norm_u = norm(u_vec)
                norm_v = norm(v_vec)
                if norm_u == 0 or norm_v == 0:
                    sim = 0
                else:
                    sim = dot / (norm_u * norm_v)
                similarities[i, j] = sim
    
    return similarities

def calculate_mean(train_df):
    # Compute the mean rating for each user
    train_array = train_df.values
    user_means = []
    num_users = train_array.shape[0]
    for i in range(num_users):
        user_ratings = train_array[i, :]
        nonzero_ratings = user_ratings[user_ratings != 0]
        if len(nonzero_ratings) > 0:
            mean_rating = nonzero_ratings.mean()
        else:
            mean_rating = 0
        user_means.append(mean_rating)

    user_means = np.array(user_means)
    return user_means

def predict(train_df, test_df, similarities, user_means):
    # compute prediction for test set
    train_array = train_df.values 
    test_array  = test_df.values 
    num_users   = train_array.shape[0]
    num_items   = train_array.shape[1]

    predictions = np.zeros_like(test_array)

    for i in range(num_users):
        for j in range(num_items):
            if test_array[i, j] != 0:
                num = 0
                den = 0
                for neighbor in range(num_users):
                    if neighbor == i:
                        continue
                    if train_array[neighbor, j] != 0:
                        sim_val = similarities[i, neighbor]
                        neighbor_mean = user_means[neighbor]
                        r_ni = train_array[neighbor, j]

                        num += sim_val * (r_ni - neighbor_mean)
                        den += abs(sim_val)

                if den > 0:
                    predictions[i, j] = user_means[i] + (num / den)
                else:
                    predictions[i, j] = user_means[i]
    return predictions

def predict_knn(train_df, test_df, similarities, user_means):
    train_array = train_df.values
    test_array  = test_df.values
    num_users   = train_array.shape[0]
    num_items   = train_array.shape[1]

    best_rmse   = float('inf')
    best_k      = 0
    best_predictions = None

    for k in range(10, 300, 10):
        predictions_k = np.zeros_like(test_array)

        for u in range(num_users):
            neighbors_sorted = np.argsort(-similarities[u])
            neighbors_sorted = neighbors_sorted[neighbors_sorted != u]
            neighbors_top_k = neighbors_sorted[:k] 

            for i in range(num_items):
                if test_array[u, i] != 0:
                    num = 0.0
                    den = 0.0
                    for neighbor in neighbors_top_k:
                        if train_array[neighbor, i] != 0:
                            sim_val = similarities[u, neighbor]
                            neighbor_mean = user_means[neighbor]
                            r_ni = train_array[neighbor, i]

                            num += sim_val * (r_ni - neighbor_mean)
                            den += abs(sim_val)

                    if den > 0:
                        predictions_k[u, i] = user_means[u] + (num / den)
                    else:
                        predictions_k[u, i] = user_means[u]

        current_rmse = cal_rmse(test_array, predictions_k)

        print(f"k = {k}, RMSE = {current_rmse:.4f}")
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_k = k
            best_predictions = predictions_k.copy()

    print(f"\nBest k found: {best_k} with RMSE {best_rmse:.4f}")
    return best_predictions

def recommend_top_movies(user_id, predictions, train_df, top_n=10):
    user_index = user_id - 1

    user_predictions = predictions[user_index]

    rated_movies = train_df.iloc[user_index]
    rated_movies = rated_movies[rated_movies > 0].index.tolist()

    predicted_ratings = pd.DataFrame({
        'title': train_df.columns,
        'predicted_rating': user_predictions
    })

    predicted_ratings = predicted_ratings[~predicted_ratings['title'].isin(rated_movies)]

    top_movies = predicted_ratings.sort_values(by='predicted_rating', ascending=False).head(top_n)

    return top_movies['title'].tolist()

def main():
    if not os.path.exists('model_knn.pkl'):
        train_df, test_df, items = load_data()
        simularity = calculate_similarity(train_df)
        user_means = calculate_mean(train_df)
        predictions = predict_knn(train_df, test_df, simularity, user_means)
        save_model(predictions, 'model_knn.pkl')
    else:
        with open('model_knn.pkl', 'rb') as f:
            predictions = pickle.load(f)
        train_df, test_df, items = load_data()
        num_users = train_df.shape[0]
        test_array = test_df.values 
        rmse  = cal_rmse(test_array, predictions)
    
    print("\nWelcome to the Movie Recommendation System 1 - User Based Collaborative Filtering")

    while True:
        try:
            user_id = int(input("Enter User ID to get recommendations: "))

            if user_id < 1 or user_id > num_users:
                print(f"Invalid User ID. Please enter a value between 1 and {num_users}.")
                continue

            top_10_movies = recommend_top_movies(user_id, predictions, train_df,top_n=10)

            print(f"\nTop 10 recommended movies for User {user_id}:")
            for i, movie in enumerate(top_10_movies, start=1):
                print(f"{i}. {movie}")

            continue_prompt = input("\nDo you want to recommend for another user? (y/n): ").strip().lower()
            if continue_prompt != 'y':
                print("Exiting the recommendation system 1.")
                break

        except ValueError:
            print("Invalid input. Please enter a valid User ID.")

if __name__ == "__main__":
    main()