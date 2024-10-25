# Movie Recommendation System

This project implements a **recommender system** using a collaborative filtering method based on **matrix factorization**. The system is designed to predict user ratings for items using the **MovieLens dataset** and the **ALS (Alternating Least Squares)** algorithm.

## Overview

The key approach used in this project is the **ALS** method, a matrix factorization technique that models latent factors for both users and items. It iteratively optimizes user and item matrices to predict missing ratings by minimizing the difference between actual ratings and predicted ones.

### Objective Function

The ALS method minimizes the following objective function:

$$
\min_{U, V} \sum_{(i,j) \in \mathcal{R}} (r_{ij} - U_i^T V_j)^2 + \lambda (||U||^2 + ||V||^2)
$$

Where:
- \( U \in \mathbb{R}^{n \times k} \) is the matrix of user latent factors.
- \( V \in \mathbb{R}^{m \times k} \) is the matrix of item latent factors.
- \( r_{ij} \) represents the observed rating by user \( i \) for item \( j \).
- \( \lambda \) is a regularization term to prevent overfitting.

The algorithm alternates between fixing \( V \) to solve for \( U \), and fixing \( U \) to solve for \( V \), ensuring efficient convergence.

## Features

- **Matrix Factorization using ALS**: A robust collaborative filtering method that scales well with large datasets.
- **MovieLens Dataset**: Utilizes the well-known [MovieLens dataset](https://grouplens.org/datasets/movielens/) for training and evaluation.
- **Performance Metrics**: Uses Root Mean Squared Error (RMSE) for model evaluation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hjisaac/recommender-system.git
   cd recommender-system
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Download Dataset**: Obtain the MovieLens dataset from [here](https://grouplens.org/datasets/movielens/) and place it in the `data/` directory.

2. **Run the Model**:
   ```bash
   python train_als.py
   ```

3. **Evaluate Results**: The script will output RMSE values for both training and test sets, indicating model performance.

## Dataset

The [MovieLens dataset](https://grouplens.org/datasets/movielens/) provides user ratings for items, enabling the training of recommendation models. The dataset includes:
- **User ID**: Identifies the user.
- **Item ID**: Identifies the item.
- **Rating**: Rating given by the user (typically on a 1-5 scale).
- **Timestamp**: Time when the rating was submitted.

## Model Training

The system learns latent features for both users and items by minimizing the regularized least squares objective. RMSE is used to evaluate the model:

$$
RMSE = \sqrt{\frac{1}{n} \sum_{(i,j) \in \mathcal{R}} (r_{ij} - \hat{r}_{ij})^2}
$$

Where \( r_{ij} \) is the actual rating, and \( \hat{r}_{ij} = U_i^T V_j \) is the predicted rating.

## Hyperparameters

- **Latent Factors**: The number of features to learn for users and items.
- **Regularization (lambda)**: Controls the complexity of the model by penalizing large weights.
- **Epochs**: Number of iterations over the data during training.

## Results

Example output for training and testing might look like:

- **Training RMSE**: 0.85
- **Test RMSE**: 0.90

These values help assess how well the model generalizes to unseen data.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contributing

Contributions are welcome! Please submit pull requests or issues if you have suggestions for improvement.

---

This repository provides a foundational implementation of a recommendation system using ALS matrix factorization techniques, demonstrated through the MovieLens dataset but applicable to other datasets as well.