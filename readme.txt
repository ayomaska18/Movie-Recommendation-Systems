```markdown
# Movie Recommendation Systems 

This repository contains two movie recommendation systems developed using the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/). It includes both traditional user-based collaborative filtering and a deep learning-based neural collaborative filtering model.

---

## Project Structure

| File            | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `rs1.py`        | User-Based Collaborative Filtering with cosine similarity                   |
| `rs2.py`        | Neural Collaborative Filtering (NCF) using PyTorch                          |
| `run_eval.py`   | Evaluation script computing RMSE and novelty for both systems               |
| `eda.ipynb`     | Jupyter Notebook for Exploratory Data Analysis (EDA)                        |

---

## Setup Instructions

### 1. Download the Dataset

Download the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/) and place the extracted `ml-100k` folder in the project root:

```

project/
├── ml-100k/
│   ├── u.data
│   ├── u.item
│   ├── u.user
│   ├── u.genre
├── rs1.py
├── rs2.py
├── run\_eval.py
├── eda.ipynb
└── README.md

````

---

## How to Run

### RS1: User-Based Collaborative Filtering
```bash
python rs1.py
````

* Computes cosine similarity between users
* Uses top-K nearest neighbors for prediction
* Prompts user input and returns top 10 movie recommendations

### RS2: Neural Collaborative Filtering (NCF)

```bash
python rs2.py
```

* Trains a PyTorch model with user/movie embeddings and MLP layers
* Saves best model based on RMSE
* Recommends top 10 movies based on learned user preferences

---

## Evaluation

Run the following to compare the two systems:

```bash
python run_eval.py --num_users 10
```

Metrics evaluated:

* **RMSE**: Accuracy of predicted ratings
* **Novelty**: Encourages recommendations beyond the most popular items

---

## Models Used

### RS1

* Collaborative filtering with cosine similarity
* Dynamic K-selection for neighbors

### RS2

* Neural network with embedding layers for users and items
* Hidden layers with dropout, batch normalization, and LeakyReLU

---

## 📈 EDA

The `eda.ipynb` notebook includes:

* Distribution of ratings
* User/movie interaction statistics
* Insights on data sparsity

---

## Demo

Once a model is trained, the system interactively asks for a `user_id`:

```
Enter User ID to get recommendations:
```

And returns:

```
Top 10 recommended movies for User X:
1. Movie A
2. Movie B
...
```

---

Install Required Libraries:

```bash
pip install -r requirements.txt
```

---

```

---
