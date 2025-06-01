import argparse
import pickle
import os
import numpy as np
import pandas as pd
from math import sqrt, log
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from rs2 import NCF
from rs1 import load_data as load_data_rs1

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_rmse_rs1(test_arr, pred):
    arr = test_arr.values
    se = 0.0
    count = 0
    num_users, num_items = arr.shape
    for r in range(num_users):
        for c in range(num_items):
            actual = arr[r, c]
            if actual > 0:
                diff = pred[r, c] - actual
                se += diff * diff
                count += 1
    return sqrt(se / count) if count else 0

def compute_rmse_rs2(model, X_test, y_test, batch_size=256):
    class MovieLensDataset(Dataset):
        def __init__(self, X, y):
            self.users = X[:, 0]
            self.items = X[:, 1]
            self.ratings = y
        def __len__(self):
            return len(self.ratings)
        def __getitem__(self, idx):
            return (torch.tensor(self.users[idx], dtype=torch.long),
                    torch.tensor(self.items[idx], dtype=torch.long),
                    torch.tensor(self.ratings[idx], dtype=torch.float))
    dataset = MovieLensDataset(X_test, y_test)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    total_se = 0.0
    total_count = 0
    with torch.no_grad():
        for user_ids, item_ids, ratings in loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)
            preds = model(user_ids, item_ids)
            se = ((preds - ratings) ** 2).sum().item()
            total_se += se
            total_count += len(ratings)
    mse = total_se / total_count
    return sqrt(mse)

def compute_novelty(recommend_func, rec_system, user_list, train_df, **kwargs):
    popularity = (train_df > 0).sum(axis=0)
    total_ratings = (train_df > 0).sum().sum()
    novelty_scores = {}
    for title, count in popularity.items():
        p = count / total_ratings if total_ratings > 0 else 0.0001
        novelty_scores[title] = -np.log2(p) if p > 0 else 0
    novelty_list = []
    for user_id in user_list:
        recommended = recommend_func(user_id, rec_system,train_df, **kwargs)
        if not recommended:
            continue
        scores = [novelty_scores.get(title, 0) for title in recommended]
        novelty_list.append(np.mean(scores))
    return np.mean(novelty_list) if novelty_list else 0

def recommend_rs1(user_id, predictions, train_df, top_n=10):
    user_index = user_id - 1
    user_predictions = predictions[user_index]
    rated_movies = train_df.iloc[user_index]
    seen_movies = rated_movies[rated_movies > 0].index.tolist()
    pred_df = pd.DataFrame({
        'title': train_df.columns,
        'predicted_rating': user_predictions
    })
    pred_df = pred_df[~pred_df['title'].isin(seen_movies)]
    top_movies = pred_df.sort_values(by='predicted_rating', ascending=False).head(top_n)
    return top_movies['title'].tolist()

def recommend_rs2(user_id, model, movie_df, item_enc, train_df, top_n=10):
    rated_movies = train_df.loc[user_id]
    seen_movies = rated_movies[rated_movies > 0].index.tolist()
    candidate_movies = movie_df[~movie_df['title'].isin(seen_movies)].copy()
    candidate_movies['encoded'] = item_enc.transform(candidate_movies['title'])
    all_movie_indices = torch.tensor(candidate_movies['encoded'].values, dtype=torch.long).to(device)
    user_indices = torch.tensor([user_id - 1] * len(candidate_movies), dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        preds = model(user_indices, all_movie_indices)
    candidate_movies['predicted_rating'] = preds.cpu().numpy()
    recommended = candidate_movies.sort_values(by='predicted_rating', ascending=False)
    return recommended['title'].head(top_n).tolist()

def load_data_rs2():
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', 
                          names=['user_id', 'movie_id', 'rating', 'timestamp'])
    ratings.drop(columns=["timestamp"], inplace=True)
    
    user = pd.read_csv('ml-100k/u.user', sep='|', 
                       names=['user_id', 'age', 'occupation', 'zip_code'])
    movie_genres = pd.read_csv('ml-100k/u.genre', sep='|', 
                               names=['genre', 'genre_id'])
    genres = list(movie_genres['genre'])
    
    columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL'] + genres
    items = pd.read_csv('ml-100k/u.item', sep='|', header=None,
                        names=columns, encoding='latin-1')
    movie_df = items[['movie_id', 'title']]

    df = pd.merge(ratings, movie_df, how='inner', on='movie_id')
    df = df.groupby(by=['user_id','title'], as_index=False).agg({"rating": "mean"})

    user_enc = LabelEncoder()
    df['user'] = user_enc.fit_transform(df['user_id'].values)
    n_users = df['user'].nunique()
    
    item_enc = LabelEncoder()
    df['movie'] = item_enc.fit_transform(df['title'].values)
    n_movies = df['movie'].nunique()
    
    X = df[['user', 'movie']].values
    y = df['rating'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=50)
    
    return X_train, X_test, y_train, y_test, n_users, n_movies, df, movie_df, user_enc, item_enc

def normalize_novelty(raw_novelty, max_possible_novelty):
    """Normalize novelty to 0-100 scale"""
    return min(100, (raw_novelty / max_possible_novelty) * 100)

def main():
    parser = argparse.ArgumentParser(description="Evaluate RS1 and RS2")
    parser.add_argument('--num_users', type=int, default=10,
                        help="Number of users to evaluate novelty on (default: 10)")
    args = parser.parse_args()

    # Load Data and Pre-trained Models
    train_df_rs1, test_df_rs1, items_rs1 = load_data_rs1()
    
    if os.path.exists('model_knn.pkl'):
        with open('model_knn.pkl', 'rb') as f:
            predictions_rs1 = pickle.load(f)
    else:
        print("RS1 predictions file 'model_knn.pkl' not found!")
        predictions_rs1 = None

    X_train, X_test, y_train, y_test, n_users, n_movies, df_rs2, movie_df, user_enc, item_enc = load_data_rs2()
    
    if os.path.exists('best_ncf_model.pkl'):
        with open('best_ncf_model.pkl', 'rb') as f:
            model_rs2 = pickle.load(f)
    else:
        print("RS2 model file 'best_ncf_model.pkl' not found!")
        model_rs2 = None
    
    # RS1 RMSE
    if predictions_rs1 is not None:
        rmse_rs1 = compute_rmse_rs1(test_df_rs1, predictions_rs1)
    else:
        rmse_rs1 = None
    
    # RS2 RMSE
    if model_rs2 is not None:
        rmse_rs2 = compute_rmse_rs2(model_rs2, X_test, y_test)
    else:
        rmse_rs2 = None
    
    # Novelty Evaluation
    user_list = list(range(1, args.num_users + 1))
    novelty_rs1 = compute_novelty(recommend_rs1, predictions_rs1, user_list, train_df_rs1, top_n=10) if predictions_rs1 is not None else None
    novelty_rs2 = compute_novelty(lambda user_id, rec_system, df: recommend_rs2(user_id, rec_system, movie_df, item_enc, df, top_n=10), model_rs2, user_list, train_df_rs1) if model_rs2 is not None else None

    all_popularities = (train_df_rs1 > 0).sum(axis=0)
    total_ratings = (train_df_rs1 > 0).sum().sum()
    min_popularity = all_popularities.min() / total_ratings
    max_possible_novelty = -np.log2(min_popularity)

    # Then normalize your scores
    rs1_normalized = normalize_novelty(novelty_rs1, max_possible_novelty)
    rs2_normalized = normalize_novelty(novelty_rs2, max_possible_novelty)
    
    # Results
    print("Evaluation Results:")
    if rmse_rs1 is not None:
        print(f"RS1 (User-Based CF) RMSE: {rmse_rs1:.4f}")
    else:
        print("RS1 RMSE: Not available")
    
    if rmse_rs2 is not None:
        print(f"RS2 (Neural CF) RMSE: {rmse_rs2:.4f}")
    else:
        print("RS2 RMSE: Not available")
    
    if novelty_rs1 is not None:
        print(f"RS1 Normalized Novelty (0-100): {rs1_normalized:.4f}")
    else:
        print("RS1 Novelty: Not available")
    
    if novelty_rs2 is not None:
        print(f"RS2 Normalized Novelty (0-100): {rs2_normalized:.4f}")
    else:
        print("RS2 Novelty: Not available")

if __name__ == "__main__":
    main()
