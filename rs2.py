import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

def evaluate_mse(model, loader):
    model.eval()
    total_se = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            user_ids, item_ids, ratings = [t.to(device) for t in batch]
            preds = model(user_ids, item_ids)
            se = ((preds - ratings)**2).sum().item()
            total_se += se
            total_count += len(ratings)
    mse = total_se / total_count
    return mse

def load_data():
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    ratings.drop(columns=["timestamp"], inplace=True)

    user = pd.read_csv('ml-100k/u.user', sep='|', names=['user_id', 'age', 'occupation', 'zip_code'])
    movie_genres = pd.read_csv('ml-100k/u.genre', sep='|', names=['genre', 'genre_id'])
    genres = list(movie_genres['genre'])

    columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL'] + genres
    items = pd.read_csv('ml-100k/u.item', sep='|', header=None, names=columns, encoding='latin-1')
    movie_df = items[['movie_id','title']]

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
    return X_train, X_test, y_train, y_test, n_users, n_movies, df, movie_df, user_enc, item_enc

class MovieLensDataset(Dataset):
    def __init__(self, X, y):
        self.users = X[:, 0]
        self.items = X[:, 1]
        self.ratings = y

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.ratings[idx], dtype=torch.float),
        )
    
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=8, hidden_dims=[64,32,16], dropout_rate=0.2):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        
        self.gmf = nn.Linear(embed_dim, 1)

        mlp_layers = []
        input_dim = 2 * embed_dim
        for hdim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(input_dim, hdim),
                nn.BatchNorm1d(hdim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hdim
        
        self.mlp = nn.Sequential(*mlp_layers)
        self.mlp_out = nn.Linear(hidden_dims[-1], 1)
        self.output = nn.Linear(2, 1)

        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.item_embed.weight)

    def forward(self, user_ids, item_ids):
        user_vec = self.user_embed(user_ids)
        item_vec = self.item_embed(item_ids)
        
        gmf_product = user_vec * item_vec
        gmf_out = self.gmf(gmf_product)
        
        mlp_input = torch.cat([user_vec, item_vec], dim=1)
        mlp_out = self.mlp_out(self.mlp(mlp_input))
        
        combined = torch.cat([gmf_out, mlp_out], dim=1)
        return self.output(combined).squeeze()

def train_ncf_model(X_train, X_test, y_train, y_test, n_users, n_movies):
    train_dataset = MovieLensDataset(X_train, y_train)
    test_dataset = MovieLensDataset(X_test, y_test)

    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = NCF(
        num_users=n_users, 
        num_items=n_movies, 
        embed_dim=4, 
        hidden_dims=[32, 16],  
        dropout_rate=0.1  
    ).to(device)


    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-3)
    epochs = 70
    train_rmse_list = []
    test_rmse_list = []
    best_rmse = float('inf')
    best_model = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            user_ids, item_ids, ratings = [t.to(device) for t in batch]
            optimizer.zero_grad()
            preds = model(user_ids, item_ids)
            loss = criterion(preds, ratings)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(ratings)

        train_mse = running_loss / len(train_dataset)
        train_rmse = np.sqrt(train_mse)
        train_rmse_list.append(train_rmse)

        test_mse = evaluate_mse(model, test_loader)
        test_rmse = np.sqrt(test_mse)
        test_rmse_list.append(test_rmse)

        if test_rmse < best_rmse:
          best_rmse = test_rmse
          best_model = model
          with open('best_ncf_model.pkl', 'wb') as f:
              pickle.dump(best_model, f)

        print(f"Epoch {epoch}/{epochs}, Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_rmse_list)+1), train_rmse_list, label='Train RMSE')
    plt.plot(range(1, len(test_rmse_list)+1), test_rmse_list, label='Test RMSE')
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Training and Test RMSE Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model

def recommend_top_movies(user_id, model, movie_df, item_enc, train_df, top_n=10):
    rated_movies = train_df.loc[user_id]
    seen_movies = rated_movies[rated_movies > 0].index.tolist()
    
    # Exclude movies already rated.
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


def main():
    if not os.path.exists('best_ncf_model.pkl'):
        X_train, X_test, y_train, y_test, n_users, n_movies = load_data()
        final_model = train_ncf_model(X_train, X_test, y_train, y_test, n_users, n_movies)
        save_model(final_model, 'best_ncf_model.pkl')
    else:
        with open('best_ncf_model.pkl', 'rb') as f:
            final_model = pickle.load(f)
        X_train, X_test, y_train, y_test, n_users, n_movies, df, movie_df, user_enc, item_enc = load_data()

        train_df = df.pivot(index='user_id', columns='title', values='rating').fillna(0)
        num_users = train_df.shape[0]
    
    print("\nWelcome to the Movie Recommendation System 2 - Neural Collaborative Filtering")

    while True:
        try:
            user_id = int(input("Enter User ID to get recommendations: "))

            if user_id < 1 or user_id > num_users:
                print(f"Invalid User ID. Please enter a value between 1 and {num_users}.")
                continue

            top_10_movies = recommend_top_movies(user_id, final_model, movie_df, item_enc, train_df, top_n=10)  

            print(f"\nTop 10 recommended movies for User {user_id}:")
            for i, movie in enumerate(top_10_movies, start=1):
                print(f"{i}. {movie}")

            continue_prompt = input("\nDo you want to recommend for another user? (y/n): ").strip().lower()
            if continue_prompt != 'y':
                print("Exiting the recommendation system 2.")
                break

        except ValueError:
            print("Invalid input. Please enter a valid User ID.")

if __name__ == "__main__":
    main()
