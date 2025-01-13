This project demonstrates the development of a **recommender system** leveraging **matrix factorization** techniques. The **Alternating Least Squares (ALS)** algorithm is at the core of this system, enabling precise user-item rating predictions. It has been validated using the **MovieLens dataset**, showcasing good performance and adaptability to real-world scenarios.

---

# Recommender System Codebase

## Table of Contents

- [Recommender System Codebase](#recommender-system-codebase)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Key Features](#key-features)
  - [Algorithms](#algorithms)
    - [Alternating Least Squares (ALS)](#alternating-least-squares-als)
      - [Workflow](#workflow)
      - [Advantages](#advantages)
  - [Datasets](#datasets)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Performance](#performance)
    - [Results (Example)](#results-example)
  - [Code Structure](#code-structure)
  - [Limitations](#limitations)
  - [Improvements](#improvements)
  - [Resources](#resources)
  - [License](#license)
  - [Feedbacks](#feedbacks)

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

- Without item features modeled:
  
$$
\min_{U, V} \lambda \sum_{(i,j) \in \mathcal{R}} (r_{ij} - (U_i^T V_j + b^{(u)}_i + b^{(v)}_j))^2 + \tau (\|U\|^2 + \|V\|^2) + \gamma (\|b^{(u)}\|^2 + \|b^{(v)}\|^2)
$$

- With item features modeled:

$$
\min_{U, V} \lambda \sum_{(i,j) \in \mathcal{R}} (r_{ij} - (U_i^T V_j + b^{(u)}_i + b^{(v)}_j))^2 + \tau (\|U\|^2 + \|V\|^2) + \gamma (\|b^{(u)}\|^2 + \|b^{(v)}\|^2)
$$

Where:
- $U$: Matrix of user latent factors $n \times k$.
- $V$: Matrix of item latent factors $m \times k$.
- $F$: Matrix of feature (when item features are also modeled) 
- $b^{(u)}$: Matrix of the user biases $1 \times k$.
- $b^{(v)}$: Matrix of the item biases $1 \times k$.
- $r_{ij}$: Observed rating for user $i$ and item $j$.
- $\lambda$: Regularization parameters accounting for the prediction residuals
- $\tau$: Regularization parameters accounting for $U$ and $V$
- $\gamma$: Regularization parameters accounting for $b^{(u)}$ and $b^{(v)}$

#### Workflow

1. Solve the optimization problem for $b^{(u)}$ keeping all the other matrices (.i.e $U$, $V$, $b^{(v)}$) fixed.

   $$b^{(u)}_i = \frac{\lambda \sum_{j \in \Omega(i)} \left( r_{ij} - \left( u_i^T v_j + b_j^{(v)} \right) \right)}{\lambda |\Omega(i)| + \gamma}$$
2. Solve the optimization problem for $U$ keeping all the other matrices fixed.

   $$ u_i = $$
3. Solve the optimization problem for $b^{(v)}$ keeping all the other matrices fixed.

   $$ b^{(v)}_j = $$
4. Solve the optimization problem for $V$ keeping all the other matrices fixed.
   - When F is modeled:
      $$ v_j = $$
   - When is not modeled:
     $$ v_j = $$
   
5. (When F modeled) Solve the optimization problem for $F$ keeping all the other matrices fixed.
   $$f_{\ell} = \frac{\sum_{n=1}^{N} \frac{N}{n \sqrt{F_{n}}} X_{n} - \left(\sum_{t=1}^{\ell} f_{t}\right) \sum_{n=1}^{N} \frac{1}{F_{n}}}{\sqrt{\sum_{n=1}^{N} \left(1 + \sum_{i=1}^{n} \frac{1}{F_{i}}\right)}}$$
6. Repeat until convergence.


#### Advantages
- Scalable to large datasets.
- Support for parallelization for computation performance.
- Handles sparsity in user-item interaction matrices effectively.

---

## Datasets

The folder `dataset` contains information about examples' dataset.

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

3. **Run an example**:
   To run the movielens example, download the dataset from [here](https://grouplens.org/datasets/movielens/).
   Ideally, put that dataset in the example folder and change the path of rating.csv file passed to the indexer.
   And run `poetry run python examples/path_to_example_file.py`
   
---  

## Usage

Only collaborative filtering is implemented now, and it is encaspsulated in the class `CollaborativeFilteringRecommenderBuilder`.

   ```py
   from src.recommenders import CollaborativeFilteringRecommenderBuilder
   # ...
   # Create everything needed instance the builder (indexed_data, backend that will run the proper algorithm..)
   # ...
   # Instantiate the builder with all the necessary arguments
   recommander_builder = CollaborativeFilteringRecommenderBuilder(*args, *kwars)  
   
   # Build the recommender now by calling the build on the builder to get the recommender (Kinda an implementation of the builder design pattern).
   # This will basically train the recommendation model, so it will take some time depending on the dataset size and the parameters.
   recommender = recommander_builder.build(*args, **kwargs)
   
   # To recommend, call the recommend method of the recommender object with a list of rating. E.g: [(item1, rating1), ..]
   recommender.recommend(input_ratings)
   
   # If called without arguments, the recommender will recommend best rated items.
   recommender.recommend()
   ```

   The script outputs RMSE and Loss values for both training and testing, providing insight into the system's predictive accuracy.
    And those values can be accessed later from the model and be plotted using the graphing utils if the model has been saved 
    (to save the model as checkpoint, one can pass `save_checkpoint=True` to the backend object used to do the training). There are 
   also some logs that will be generated in the `artifacts/logs` folder each time the backend runs. Those logs can be very usefull
    for debugging purposes.

---

## Performance

The first example implemented to access performance uses the [**MovieLens dataset**](https://grouplens.org/datasets/movielens/).

**Root Mean Squared Error (RMSE)** is used as the primary evaluation metric:

$$
RMSE = \sqrt{\frac{1}{n} \sum_{(i,j) \in \mathcal{R}} (r_{ij} - \hat{r}_{ij})^2}
$$

Where:
- $r_{ij}$: Actual rating.
- $\hat{r}_{ij} = U_i^T V_j + b^{(u)}_i + b^{(v)}_j$: Predicted rating.

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
│       ├── 1000000   # Checkpoint for ALS with 1 million interactions as limit of lines to load.
│       └── 100000000 # Checkpoint for ALS with 100 million interactions as limit of lines to load.
├── figures/          # Contains visualizations or figures generated during the project (for analysis and results...).
└── logs/             # Logging files generated during training or testing.

datasets/             # Documentation about the datasets used for training and evaluation of the recommender system.

docs/                 # Documentation for the project, including detailed explanations and guidelines.

examples/             # Example scripts to demonstrate the usage of the system.
├── basic_example/    # A simple example to get started quickly.
└── movies_lens/      # Example using the MovieLens dataset.

figures/              # Additional plots and figures for analysis and results.

src/                  # Source code for the project, organized by functional modules.
├── algorithms/       # Implementation of recommender system algorithms.
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

## Limitations 

Running the MovieLens example (32 million ratings) takes approximately 3 hours on CPU alone.

The early use of `SerialUnidirectionalMapper` and `SerialBidirectionalMapper` data structures 
complicates integrating [Numba](https://numba.pydata.org/). These classes lack clear type specifications, making it nearly 
impossible to leverage **Numba**'s optimization capabilities. To use Numba effectively, we would 
need to remove these data structures from the code. There is an issue opened to fix that.

## Improvements

- **Integration of Additional Algorithms**: Incorporate other collaborative filtering and content-based methods.
- **Hybrid Recommender Systems**: Combine collaborative and content-based filtering for improved performance.
- **More Examples**: Implement more examples potentially with different datasets for real-time recommendations.
- **Numba and Jax**: Currently it is not possible to use numba/jax, cause the code contains a lot of custom objects. Plan a fix.
- **Issues**: Resolve the remaining issues, including the adding of unit tests.
- **Comparison**: Compare with existing libraries.
---

## Resources

The `docs` folder contains useful resources (papers, ...)

---
## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Feedbacks

Feel free to give any feedback or report any issues to me [<hjisaac.h at gmail.com>](hjisaac.h@gmail.com). 





