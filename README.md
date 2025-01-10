??
---

# Recommendation System Building (Framework)

This project demonstrates the development of a **high-performing recommender system** leveraging **matrix factorization** techniques. The **Alternating Least Squares (ALS)** algorithm is at the core of this system, enabling precise user-item rating predictions. It has been extensively validated using the **MovieLens dataset**, showcasing exceptional performance and adaptability to real-world scenarios.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Algorithms](#algorithms)
   - [Alternating Least Squares (ALS)](#alternating-least-squares-als)
4. [Dataset](#dataset)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Performance](#performance)
8. [Code Structure](#code-structure)
9. [Improvements](#improvements)
10. [License](#license)

---

## Introduction

This project is designed to deliver **accurate and scalable recommendations** using collaborative filtering. The **ALS algorithm** powers the system, offering strong generalization and robust prediction capabilities.

The codebase is crafted with a focus on **modularity** and **reusability**, ensuring it serves as a foundation for both research and real-world applications.

---

## Key Features

- **State-of-the-Art Algorithm**: Implements the ALS matrix factorization algorithm
- **Scalable Design**: Handles datasets with millions of user-item interactions, ensuring practical usability.
- **Performance Validation**: Extensively tested on the **MovieLens dataset**, achieving excellent prediction accuracy.
- **Extensibility**: The modular architecture supports easy integration of additional algorithms or datasets.

---

## Algorithms

### Alternating Least Squares (ALS)

**ALS** is a collaborative filtering technique based on **matrix factorization**. It models user and item interactions by discovering latent features that explain observed ratings. The algorithm alternates between optimizing user and item matrices to minimize the regularized objective function:

$$
\min_{U, V} \lambda \sum_{(i,j) \in \mathcal{R}} (r_{ij} - (U_i^T V_j + b^{(u)}_i + b^{(v)}_j))^2 + \tau (\|U\|^2 + \|V\|^2) + \gamma (\|b^{(u)}\|^2 + \|b^{(v)}\|^2)
$$

Where:
- $U$: Matrix of user latent factors $n \times k$.
- $V$: Matrix of item latent factors $m \times k$.
- $b^{(u)}$: Matrix of the user biases $1 \times k$.
- $b^{(v)}$: Matrix of the item biases $1 \times k$.
- $r_{ij}$: Observed rating for user $i$ and item $j$.
- $\lambda$: Regularization parameters accounting for the prediction residuals
- $\tau$: Regularization parameters accounting for $U$ and $V$
- $\gamma$: Regularization parameters accounting for $b^{(u)}$ and $b^{(v)}$

#### Workflow:
1. Solve the optimization problem for $b^{(u)}$ keeping all the other matrices (.i.e $U$, $V$, $b^{(v)}$) fix.
      $$ b^{(u)} = $$
2. Solve the optimization problem for $U$ keeping all the other matrices fix.
3. Solve the optimization problem for $b^{(v)}$ keeping all the other matrices fix.
4. Solve the optimization problem for $V$ keeping all the other matrices fix.
5. Repeat until convergence.

#### Advantages:
- Scalable to large datasets.
- Support for parallelism
- Handles sparsity in user-item interaction matrices effectively.

---

## Datasets
The folder `dataset` contains information about examples' dataset.

### MovieLens dataset
The first example implemented to access performance uses the [**MovieLens dataset**](https://grouplens.org/datasets/movielens/).
This dataset is a standard benchmark for evaluating recommendation algorithms.

---

## Installation

To set up the project:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hjisaac/recommender-system.git
   cd recommender-system
   ```

2. **Install dependencies**:
   ```bash
   poetry install
   ```

3. **Download the dataset**:
   To run the movielenz example, download the dataset from [here](https://grouplens.org/datasets/movielens/).

---

## Usage

1. **Run the model**:
   ```bash
   poetry run python main.py
   ```

2. **Evaluate performance**:
   The script outputs RMSE values for both training and testing, providing insight into the system's predictive accuracy.

---

## Performance

**Root Mean Squared Error (RMSE)** is used as the primary evaluation metric:

$$
RMSE = \sqrt{\frac{1}{n} \sum_{(i,j) \in \mathcal{R}} (r_{ij} - \hat{r}_{ij})^2}
$$

Where:
- $r_{ij}$: Actual rating.
- $\hat{r}_{ij}$: Predicted rating.

### Results (Example)
- **Training RMSE**: 0.84
- **Test RMSE**: 0.88

These results demonstrate the model's ability to generalize well to unseen data, confirming its practical applicability.

---

## Code Structure

```txt
artifacts/            # Stores generated artifacts such as model checkpoints, logs, and profiling data.
├── checkpoints/      # Saved model checkpoints for resuming or fine-tuning training.
│   └── als/          # Checkpoints for the ALS algorithm specifically.
│       ├── 1000000   # Checkpoint for ALS with 1 million interactions.
│       └── 100000000 # Checkpoint for ALS with 100 million interactions.
├── figures/          # Contains visualizations or figures generated during the project.
└── logs/             # Logging files generated during training or testing.

datasets/             # Datasets used for training and evaluation of the recommender system.

docs/                 # Documentation for the project, including detailed explanations and guidelines.

examples/             # Example scripts to demonstrate the usage of the system.
├── basic_example/    # A simple example to get started quickly.
└── movies_lens/      # Example using the MovieLens dataset.

figures/              # Additional plots and figures for analysis and results.

ml-32m/               # Preprocessed or raw data specifically for the MovieLens 32M dataset.

src/                  # Source code for the project, organized by functional modules.
├── algorithms/       # Core algorithms used in the recommender system.
│   └── core/         # Implementation of the base logics common to all the recommender algorithms.
├── backends/         # Backend modules for database access, API integrations, etc.
├── helpers/          # Utility functions and helpers for common tasks.
├── recommenders/     # High-level classes to encapsulate recommendation pipelines.
├── settings/         # Configuration files for the project.
└── utils/            # General-purpose utilities used throughout the codebase.

tests/                # Test suite for validating the functionality of the project.
├── backends/         # Tests specific to backend modules.
├── fixtures/         # Sample test data or configurations for consistent testing.
├── helpers/          # Tests for utility functions and helpers.
│   └── test_checkpoints/ # Tests for the checkpoint loading and saving functionality.
└── utils/            # Tests for utilities used across the codebase.


```

---

## Improvements

- **Integration of Additional Algorithms**: Incorporate other collaborative filtering and content-based methods.
- **Hybrid Recommender Systems**: Combine collaborative and content-based filtering for improved performance.
- **More Examples**: Implement more examples potentially with different datasets for real-time recommendations.
- **Numba and Jax**: Currently it is not possible to use numba/jax, cause the code contains a lot of custom objects. Plan a fix.
- **Issues**: Resolve the remaining issues, including the adding of unit tests.
- **Comparison**: Compare with existing libraries
---

## Resources
The `docs` folder contains useful resources (papers, ...)

---
## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.




