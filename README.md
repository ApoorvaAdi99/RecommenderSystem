# Implicit Feedback Recommendation System (Collaborative Filtering & Graph Models)

This project builds a recommendation engine for large-scale implicit user–item interaction data (Dataset 3). It generates top-20 personalized item recommendations and evaluates multiple algorithms using NDCG@20.

## Overview
- Dataset size: ~52k users, ~91k items, ~2.3M interactions  
- Data type: Implicit positive feedback  
- Goal: Recommend 20 items per user for leaderboard submission  
- Pipeline: Load → Preprocess → Train → Evaluate → Generate Submission  

## Implemented Models (with rationale)

### Item-Based Collaborative Filtering (Cosine)
Strong, reliable baseline for sparse implicit datasets.

### Item-Based CF (Jaccard)
Co-occurrence based similarity for binary interactions.

### SVD Matrix Factorization
Learns latent factors for users and items.

### Neural Collaborative Filtering
Models nonlinear user–item interaction patterns.

### Alternating Least Squares (ALS)
Scalable matrix factorization for implicit feedback.

### LightGCN
Graph-based model leveraging user–item bipartite structure.

### Ensemble
Tests whether combining models yields improvements.

**Best performing approach:** Item-Based CF (Cosine)

## Results (NDCG@20)

| Model                      | NDCG@20 |
|---------------------------|---------|
| Item-CF (Cosine)          | 0.3056  |
| Neural CF                 | 0.2876  |
| Jaccard CF                | 0.1635  |
| LightGCN                  | 0.1588  |
| SVD                       | 0.1803  |
| ALS                       | 0.0293  |
| Ensemble                  | 0.0050  |

## How to Run

### Evaluate a model
```bash
python recommender_system.py --mode evaluate --algorithm itemcf_cosine
```
