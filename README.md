# 🎬 Movie Recommendation System - Item-Based Collaborative Filtering

A comprehensive movie recommendation system implementing item-based collaborative filtering algorithm with performance evaluation and interactive user interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-red.svg)

## 🎯 Overview

This project implements an intelligent movie recommendation system using **item-based collaborative filtering** with cosine similarity. The system analyzes user rating patterns to suggest personalized movie recommendations and includes comprehensive performance evaluation metrics.

### ✨ Key Features

- 🔍 **Item-Based Collaborative Filtering** with cosine similarity
- 📊 **Performance Evaluation** using Precision@K and Recall@K metrics
- 🎮 **Interactive User Interface** for real-time recommendations
- 📈 **Comprehensive EDA** with visualization
- 🔄 **Cold-Start Handling** with popularity-based fallback
- 🎬 **MovieLens 100K Dataset** integration

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
Jupyter Notebook
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/ausaaf-rh/movie-recommendation-system-collaborative-filtering.git
cd movie-recommendation-system-collaborative-filtering
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Dataset

**MovieLens 100K Dataset**
- 📈 100,000 ratings
- 👥 943 unique users  
- 🎬 1,682 unique movies
- ⭐ Rating scale: 1-5 stars
- 🏢 Source: GroupLens Research

## 🔧 System Architecture

```
Input Layer (User Ratings) 
    ↓
Data Processing Layer (Matrix Construction)
    ↓
Algorithm Layer (Similarity Computation)
    ↓
Recommendation Layer (Score Generation)
    ↓
Evaluation Layer (Metrics Calculation)
    ↓
Interface Layer (User Interaction)
```

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| Precision@10 | 8.34% |
| Recall@10 | 7.56% |
| Users Evaluated | 655 |
| Training Time | ~2 seconds |

## 🎮 Usage Example

```python
# Generate recommendations for User ID 1
results = main_run(example_user=1, topn=10, test_size=0.2)

# Interactive recommendation system
get_user_recommendations(results)
```

**Sample Output:**
```
Top-5 Recommendations for User 1:
1. Titanic (1997) - Score: 4.23
2. Star Wars (1977) - Score: 4.15
3. Forrest Gump (1994) - Score: 4.08
```

## 🔬 Algorithm Details

### Item-Based Collaborative Filtering
- **Similarity Metric:** Cosine Similarity
- **Formula:** `sim(i,j) = (Ri · Rj) / (||Ri|| × ||Rj||)`
- **Prediction:** `P(u,i) = Σ(sim(i,j) × r(u,j)) / Σ|sim(i,j)|`

### Evaluation Methodology
- **Train/Test Split:** 80/20 random holdout
- **Metrics:** Precision@K, Recall@K
- **Baseline:** Popularity-based recommendations

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [GroupLens Research](https://grouplens.org/) for MovieLens dataset
- scikit-learn community for machine learning tools
- Academic research in collaborative filtering algorithms

## 📚 References

1. Sarwar, B., et al. (2001). "Item-based collaborative filtering recommendation algorithms."
2. Ricci, F., et al. (2015). "Recommender Systems Handbook."
3. MovieLens Dataset Documentation

---

⭐ **Star this repository if you found it helpful!**