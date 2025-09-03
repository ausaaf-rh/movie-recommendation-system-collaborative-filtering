import os
import io
import zipfile
import urllib.request
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# -------------------- Config --------------------
DATA_DIR = "data/ml-100k"
ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
RATINGS_FILE = os.path.join(DATA_DIR, "u.data")
ITEMS_FILE = os.path.join(DATA_DIR, "u.item")
RANDOM_STATE = 42

# -------------------- Utilities --------------------
def download_ml100k_if_needed():
    """Download and extract MovieLens 100k if not present."""
    if os.path.exists(RATINGS_FILE) and os.path.exists(ITEMS_FILE):
        print("Dataset already present:", DATA_DIR)
        return
    os.makedirs("data", exist_ok=True)
    print("Downloading MovieLens 100K ...")
    try:
        with urllib.request.urlopen(ZIP_URL, timeout=60) as resp:
            data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall("data")
        if not (os.path.exists(RATINGS_FILE) and os.path.exists(ITEMS_FILE)):
            raise FileNotFoundError("Dataset not found after extraction.")
        print("Downloaded and extracted to:", DATA_DIR)
    except Exception as e:
        msg = (
            "Automatic download failed. Please download ml-100k.zip manually from:\n"
            "https://grouplens.org/datasets/movielens/100k/\n"
            "and unzip into ./data/ml-100k/"
        )
        raise RuntimeError(msg) from e

def load_data():
    """Return ratings_df, items_df"""
    cols = ["user_id", "item_id", "rating", "timestamp"]
    ratings = pd.read_csv(RATINGS_FILE, sep="\t", names=cols, engine="python")
    items = pd.read_csv(ITEMS_FILE, sep="|", header=None, encoding="latin-1")
    items = items[[0,1]]
    items.columns = ["item_id", "title"]
    items["item_id"] = items["item_id"].astype(int)
    return ratings, items

# -------------------- EDA --------------------
def quick_eda(ratings, items):
    print("Ratings shape:", ratings.shape)
    print("Unique users:", ratings['user_id'].nunique(), "Unique items:", ratings['item_id'].nunique())
    print("Rating counts:")
    print(ratings['rating'].value_counts().sort_index())

    plt.figure(figsize=(8,3))
    sns.countplot(x='rating', data=ratings, order=sorted(ratings['rating'].unique()))
    plt.title("Rating distribution")
    plt.tight_layout()
    plt.show()

    pop = ratings.groupby('item_id').agg(count=('rating','size'), avg=('rating','mean')).reset_index()
    top10 = pop.sort_values(['count','avg'], ascending=[False,False]).head(10).merge(items, on='item_id')
    print("\nTop-10 most rated movies (popularity baseline):")
    display_cols = top10[['item_id','title','count','avg']]
    print(display_cols.to_string(index=False))
    return top10

# -------------------- Build interaction matrices --------------------
def build_user_item_matrix(ratings_df, fillna=0):
    """
    Returns user-item pivot DataFrame (users as rows, items as columns).
    Missing entries filled with `fillna` (default 0).
    """
    pivot = ratings_df.pivot(index='user_id', columns='item_id', values='rating').fillna(fillna)
    return pivot

# -------------------- Item-item similarity (cosine) --------------------
def compute_item_similarity(pivot_df):
    """
    pivot_df: users x items (DataFrame)
    returns: item_similarity matrix (numpy array), item_id_list (columns of pivot_df)
    """
    item_matrix = pivot_df.T.values  # items x users
    # cosine similarity between items
    sim = cosine_similarity(item_matrix)  # shape: items x items
    return sim, list(pivot_df.columns)

# -------------------- Recommendation function --------------------
def recommend_items_for_user_knn(user_id, pivot_df, item_sim, item_ids, topn=10):
    """
    Returns DataFrame with top-n recommended item_ids (no titles).
    - Excludes items the user already rated in pivot_df.
    - Uses weighted sum of item similarities x user_ratings.
    """
    if user_id not in pivot_df.index:
        raise ValueError(f"User {user_id} not in pivot matrix (cold-start).")

    user_ratings = pivot_df.loc[user_id].values  # length = n_items
    # indices of items user rated > 0
    rated_indices = np.where(user_ratings > 0)[0]
    if len(rated_indices) == 0:
        # cold-start within known users -> return top popular by count (fallback should be applied externally)
        return []

    # score for each item = sum over rated items (similarity * rating)
    scores = np.zeros(len(item_ids), dtype=float)
    for idx in rated_indices:
        sim_row = item_sim[:, idx]
        scores += sim_row * user_ratings[idx]

    # set scores of already-rated items to -inf to avoid recommending them
    scores[rated_indices] = -np.inf

    # top indices
    top_indices = np.argsort(scores)[-topn:][::-1]
    top_item_ids = [item_ids[i] for i in top_indices]
    top_scores = [scores[i] for i in top_indices]
    return list(zip(top_item_ids, top_scores))

# -------------------- Evaluation: Precision@K / Recall@K --------------------
def evaluate_precision_recall(ratings_df, pivot_df, item_sim, item_ids, k=10, test_df=None):
    """
    Evaluate recommender using test_df (user,item) pairs.
    We compute Precision@K and Recall@K averaged across users who have at least one test item
    and at least one train item.
    - test_df: DataFrame with columns user_id,item_id (if None, use a random 20% holdout)
    """
    if test_df is None:
        # default: simple random holdout 20% of ratings, stratified by user if possible
        train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=RANDOM_STATE)
    else:
        train_df = ratings_df.drop(test_df.index, errors='ignore')  # fallback

    # We will assume pivot_df corresponds to the train split; caller must ensure that.
    user_groups = test_df.groupby('user_id')['item_id'].apply(set).to_dict()
    precisions = []
    recalls = []
    n_users_evaluated = 0

    for user, true_items in user_groups.items():
        if user not in pivot_df.index:
            continue  # user cold-start -> skip (or handle separately)
        # generate recommendations
        recs = recommend_items_for_user_knn(user, pivot_df, item_sim, item_ids, topn=k)
        if not recs:
            continue
        rec_item_ids = [iid for iid, _ in recs]
        # compute metrics
        hit_count = len(set(rec_item_ids) & set(true_items))
        prec = hit_count / k
        rec = hit_count / len(true_items) if len(true_items) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        n_users_evaluated += 1

    if n_users_evaluated == 0:
        return 0.0, 0.0, 0

    return float(np.mean(precisions)), float(np.mean(recalls)), n_users_evaluated

# -------------------- Main runnable flow --------------------
def main_run(example_user=1, topn=10, test_size=0.2):
    # 1) download if needed + load
    download_ml100k_if_needed()
    ratings, items = load_data()

    # 2) quick EDA
    top10 = quick_eda(ratings, items)

    # 3) train/test split (random holdout)
    train_df, test_df = train_test_split(ratings, test_size=test_size, random_state=RANDOM_STATE)
    print(f"\nTrain shape: {train_df.shape}, Test shape: {test_df.shape}")

    # 4) build pivot from train only
    pivot_train = build_user_item_matrix(train_df, fillna=0)

    # 5) compute item similarity from train pivot
    item_sim_matrix, item_id_list = compute_item_similarity(pivot_train)
    print("Computed item-item similarity matrix.")

    # 6) popularity baseline for cold-start fallback
    pop_counts = train_df.groupby('item_id').size().sort_values(ascending=False)
    pop_top_items = pop_counts.index[:topn].tolist()

    # 7) evaluate on test set
    p_at_10, r_at_10, users_eval = evaluate_precision_recall(train_df, pivot_train, item_sim_matrix, item_id_list, k=topn, test_df=test_df)
    print(f"\nEvaluation (Item-KNN) — Precision@{topn}: {p_at_10:.4f}, Recall@{topn}: {r_at_10:.4f} (users_eval={users_eval})")

    # 8) produce recommendations for example_user
    print(f"\nTop-{topn} recommendations for user {example_user}:")
    if example_user not in pivot_train.index:
        print("User not in train (cold-start). Returning popularity baseline items.")
        recs = [(iid, None) for iid in pop_top_items]
    else:
        recs = recommend_items_for_user_knn(example_user, pivot_train, item_sim_matrix, item_id_list, topn=topn)

    # map ids -> titles and display
    recs_df = pd.DataFrame(recs, columns=['item_id','score'])
    recs_df = recs_df.merge(items, on='item_id', how='left')
    # order by score (if score None, keep popularity order)
    recs_df['score'] = recs_df['score'].fillna(-1.0)
    recs_df = recs_df.sort_values('score', ascending=False).reset_index(drop=True)
    print(recs_df[['item_id','title','score']].to_string(index=False))

    # 9) Also print popularity baseline top-10 titles
    print("\nPopularity baseline (top-10 most rated in train):")
    pop_df = pd.DataFrame({'item_id': pop_top_items}).merge(items, on='item_id')
    print(pop_df[['item_id','title']].to_string(index=False))

    # 10) return key objects for further use
    return {
        'train_df': train_df,
        'test_df': test_df,
        'pivot_train': pivot_train,
        'item_sim_matrix': item_sim_matrix,
        'item_id_list': item_id_list,
        'recommendations_df': recs_df,
        'popularity_top10': pop_df,
        'metrics': {'precision@k': p_at_10, 'recall@k': r_at_10, 'users_evaluated': users_eval}
    }

# -------------------- Run script --------------------
if __name__ == "__main__":
    results = main_run(example_user=1, topn=10, test_size=0.2)

    # Interactive system (add this to Cell 1)
def get_user_recommendations(results):
    try:
        user_id = int(input("Enter User ID: "))
        n_recs = int(input("Number of recommendations (default 5): ") or "5")
        
        if user_id in results['pivot_train'].index:
            recs = recommend_items_for_user_knn(
                user_id, 
                results['pivot_train'], 
                results['item_sim_matrix'], 
                results['item_id_list'], 
                topn=n_recs
            )
            if recs:
                _, items = load_data()
                recs_df = pd.DataFrame(recs, columns=['item_id','score'])
                recs_df = recs_df.merge(items, on='item_id')
                print(f"\nRecommendations for User {user_id}:")
                print(recs_df[['title','score']].to_string(index=False))
            else:
                print("No recommendations found")
        else:
            print("User not found in dataset")
    except:
        print("Invalid input")

# Menu system
if 'results' in locals():
    while True:
        print("\n1. Get recommendations for user")
        print("2. Show system metrics") 
        print("3. Exit")
        choice = input("Choice: ")
        
        if choice == "1":
            get_user_recommendations(results)
        elif choice == "2":
            print(f"Precision@10: {results['metrics']['precision@k']:.4f}")
            print(f"Recall@10: {results['metrics']['recall@k']:.4f}")
        elif choice == "3":
            break
        else:
            print("Invalid choice!")