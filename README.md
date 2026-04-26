# Learn ML

A comprehensive learning resource for machine learning fundamentals, built on mathematical foundations.

## About

This repository contains interactive Jupyter notebooks covering essential mathematics concepts and ML algorithms, progressing from basic linear algebra through calculus and automatic differentiation to combinatorics and probability.

## 📚 Curriculum Overview

This repository is organized into four progressive learning tracks:

| Course | Focus | Topics | Files |
|--------|-------|--------|-------|
| **ML-101** | Linear Algebra | Matrices, vectors, transformations, eigenvalues, PCA | 12 |
| **ML-102** | Calculus & Optimization | Derivatives, gradient descent, neural networks | 12 |
| **ML-103** | Combinatorics & Probability | Counting, permutations, combinations, probability | 12+ |
| **ML-201** | Data & Features | EDA, data cleaning, feature engineering | 20+ |

### Recommended Learning Path
1. **ML-101** - Build mathematical foundations
2. **ML-102** - Learn optimization and neural networks
3. **ML-103** - Understand probability and statistics
4. **ML-201** - Apply to real-world data analysis

---

## 📚 ML-101: Linear Algebra Fundamentals

The `ml-101/` directory contains notebooks organized by topic:

### Fundamentals

- [**math_001.ipynb**](ml-101/math_001.ipynb) - Introduction and foundational concepts

### Arrays & Basic Operations

- [**math_002_array_oprations.ipynb**](ml-101/math_002_array_oprations.ipynb) - Array operations and manipulation

### Equation Solving

- [**math_003_solving_equations.ipynb**](ml-101/math_003_solving_equations.ipynb) - Solving linear equations

### Matrix Foundations

- [**math_004_matrix_operations.ipynb**](ml-101/math_004_matrix_operations.ipynb) - Basic matrix operations
- [**math_005_gaussian_elminiation.ipynb**](ml-101/math_005_gaussian_elminiation.ipynb) - Gaussian elimination method

### Linear Transformations

- [**math_006_linear_transform.ipynb**](ml-101/math_006_linear_transform.ipynb) - Linear transformation concepts
- [**math_006_linear_transformation_exercises.ipynb**](ml-101/math_006_linear_transformation_exercises.ipynb) - Practice exercises for linear transformations

### Vector & Matrix Operations

- [**math_007_dot_product.ipynb**](ml-101/math_007_dot_product.ipynb) - Dot product fundamentals
- [**math_008_multiply_matrix_vector.ipynb**](ml-101/math_008_multiply_matrix_vector.ipynb) - Matrix-vector multiplication
- [**math_009_matrix_multiplication.ipynb**](ml-101/math_009_matrix_multiplication.ipynb) - Matrix-matrix multiplication

### Matrix Inversion

- [**math_010_matrix_inverse_computation_gaussian_elimination.ipynb**](ml-101/math_010_matrix_inverse_computation_gaussian_elimination.ipynb) - Computing matrix inverse using Gaussian elimination
- [**math_010_matrix_inverse.ipynb**](ml-101/math_010_matrix_inverse.ipynb) - Matrix inverse properties and applications

### Eigenvectors & Eigenvalues

- [**math_011_eigenvectors_eigenvalues_computation.ipynb**](ml-101/math_011_eigenvectors_eigenvalues_computation.ipynb) - Computing eigenvectors and eigenvalues
- [**math_011_eigenvectors_eigenvalues.ipynb**](ml-101/math_011_eigenvectors_eigenvalues.ipynb) - Theory and applications of eigenvectors and eigenvalues

### Principal Component Analysis

- [**math_012_pca_function.ipynb**](ml-101/math_012_pca_function.ipynb) - PCA implementation and functions
- [**math_012_principal_component_analysis.ipynb**](ml-101/math_012_principal_component_analysis.ipynb) - Principal component analysis concepts and applications

## 📚 ML-102: Calculus & Optimization

The `ml-102/` directory contains advanced mathematics topics:

### Calculus & Derivatives

- [**math_001_derivatives_using_sympy.ipynb**](ml-102/math_001_derivatives_using_sympy.ipynb) - Symbolic differentiation using SymPy, covering:
  - Derivatives of exponential functions (e^x)
  - Logarithmic derivatives (ln(x))
  - Trigonometric derivatives (sin(x), cos(x))
  - Fundamental differentiation rules (sum, product, constant multiple, chain rule)

- [**math_002_numeric_derivatives_using_numpy.ipynb**](ml-102/math_002_numeric_derivatives_using_numpy.ipynb) - Numerical derivatives using NumPy, covering:
  - Finite difference methods (forward, backward, central)
  - Implementing numerical derivatives with NumPy
  - Comparing numerical vs analytical derivatives
  - Effect of step size on accuracy
  - NumPy's built-in gradient() function

- [**math_003_automatic_derivative_using_jax.ipynb**](ml-102/math_003_automatic_derivative_using_jax.ipynb) - Automatic differentiation using JAX, covering:
  - Introduction to automatic differentiation (AD)
  - Forward and reverse mode AD
  - JAX's grad() for computing gradients
  - Higher-order derivatives
  - Vectorized automatic differentiation
  - Machine learning applications (gradient descent)
  - Comparison of all three approaches (symbolic, numerical, automatic)

- [**math_004_automatic_derivatives_using_torch.ipynb**](ml-102/math_004_automatic_derivatives_using_torch.ipynb) - Automatic differentiation using PyTorch, covering:
  - PyTorch's autograd system
  - Computational graphs and backpropagation
  - Computing gradients with .backward()
  - Higher-order derivatives
  - Vectorized gradient computation
  - Gradient descent optimization
  - Machine learning examples
  - Performance comparison (PyTorch vs JAX vs SymPy vs Numerical)

- [**math_005_partial_derivatives_using_sympy.ipynb**](ml-102/math_005_partial_derivatives_using_sympy.ipynb) - Partial derivatives using SymPy, covering:
  - Understanding partial derivatives symbolically
  - Computing partial derivatives with respect to multiple variables
  - Evaluating partial derivatives at specific points using `.subs()`
  - The gradient vector and its geometric meaning
  - 3D surface and contour plot visualization
  - Higher-order and mixed partial derivatives (Hessian matrix)
  - Complex example combining product rule and chain rule ($x \cdot e^y + \sin(xy)$)

### Loss Functions

- [**math_006_squared_loss.ipynb**](ml-102/math_006_squared_loss.ipynb) - Mean Squared Error (MSE) Loss, covering:
  - Squared loss fundamentals and applications
  - Calculation for single and multiple predictions
  - Visualization of loss behavior
  - Comparison with other loss functions
  - Implementation with NumPy, JAX, PyTorch, and SymPy

- [**math_007_log_loss.ipynb**](ml-102/math_007_log_loss.ipynb) - Log Loss (Cross-Entropy Loss), covering:
  - Binary and multi-class classification
  - Mathematical foundations and properties
  - Single and multiple predictions
  - Visualization of log loss behavior
  - Implementation with NumPy, PyTorch, and SymPy
  - When to use log loss vs other loss functions
  - Real-world classification examples

### Gradient Descent

- [**math_008_gradient_descent_single_variable.ipynb**](ml-102/math_008_gradient_descent_single_variable.ipynb) - Gradient descent for a single variable, covering:
  - Minimizing $f(x) = e^x - \log(x)$ using gradient descent
  - The update rule: $x_{n+1} = x_n - \alpha \cdot f'(x_n)$
  - Effect of different learning rates on convergence
  - Effect of iteration count on accuracy
  - Visualization of optimization paths and convergence
  - Different starting points converging to the same minimum

- [**math_009_gradient_descent_linear_regression.ipynb**](ml-102/math_009_gradient_descent_linear_regression.ipynb) - Gradient descent for linear regression, covering:
  - Fitting a line by minimizing Mean Squared Error (MSE)
  - Computing gradients of MSE with respect to slope and intercept
  - Effect of learning rates on convergence of both parameters
  - Visualization of fitted lines, loss, slope, and intercept convergence
  - Comparison with the analytical solution (Normal Equations)
  - Effect of iteration count on convergence

### Neural Networks

- [**math_010_neural_network_single_perceptron_single_input.ipynb**](ml-102/math_010_neural_network_single_perceptron_single_input.ipynb) - Single perceptron with single input, covering:
  - Perceptron architecture and computation
  - Weights, bias, and activation functions
  - Forward pass and basic neural computation
  - Interactive visualization of perceptron behavior
  - Foundation for understanding neural networks

- [**math_011_neural_network_single_perceptron_multiple_input.ipynb**](ml-102/math_011_neural_network_single_perceptron_multiple_input.ipynb) - Single perceptron with multiple inputs, covering:
  - Multi-dimensional input handling
  - Dot product in neural computation
  - Decision boundaries in higher dimensions
  - Generalization to multiple features
  - Visualization of linear separability

- [**math_012_sigmoid_function.ipynb**](ml-102/math_012_sigmoid_function.ipynb) - Sigmoid activation function, covering:
  - Sigmoid function properties and applications
  - Smooth probability mapping (0 to 1)
  - Advantages over step functions
  - Derivatives and backpropagation implications
  - Integration with neural networks

## 📚 ML-103: Combinatorics & Probability

The `ml-103/` directory covers combinatorics and discrete mathematics fundamentals:

### Counting Principles

- [**math_001_counting.ipynb**](ml-103/math_001_counting.ipynb) - Fundamental counting principles, covering:
  - Basic counting rules and techniques
  - Multiplication principle and addition principle
  - Sample spaces and event counting
  - Real-world applications in probability
  - Foundation for combinatorial analysis

### Permutations

- [**math_002_permutations.ipynb**](ml-103/math_002_permutations.ipynb) - Permutations and arrangements, covering:
  - Ordered arrangements of elements
  - Factorial notation and calculations
  - Permutations with and without repetition
  - Applications to real-world problems
  - Interactive visualizations and computational methods

### Combinations

- [**math_003_combinations.ipynb**](ml-103/math_003_combinations.ipynb) - Combinations and selection, covering:
  - Unordered selection of elements
  - Binomial coefficients and Pascal's triangle
  - Combination formulas and properties
  - Classic problems: lottery, poker hands, committee selection
  - Visual demonstrations of symmetry and identities
  - Comparison with permutations
  - **Enhanced visualizations**: 3-panel layouts with intuitive graphics showing distributions, example cases, and mathematical formulas

### Stars and Bars Method

- [**math_004_stars_bars.ipynb**](ml-103/math_004_stars_bars.ipynb) - Distribution counting with stars and bars, covering:
  - Distributing identical items into distinct bins
  - Non-negative and positive integer solutions
  - Constraint transformation techniques
  - Inclusion-exclusion principle applications
  - Practical problems: sticker distribution, integer solutions, bagel selection
  - **Improved visualizations**: 3-panel layouts with colored boxes showing distributions, step-by-step explanations, and clear calculations for 6 practice problems

### Probability Theory

- [**math_006_01_probability_definition.ipynb**](ml-103/math_006_01_probability_definition.ipynb) - Fundamental probability definitions
- [**math_006_02_equally_likely_outcomes.ipynb**](ml-103/math_006_02_equally_likely_outcomes.ipynb) - Equally likely outcomes and probability
- [**math_006_03_probability_or.ipynb**](ml-103/math_006_03_probability_or.ipynb) - OR operations in probability
- [**math_006_04_conditional_probability.ipynb**](ml-103/math_006_04_conditional_probability.ipynb) - Conditional probability
- [**math_006_05_law_of_total_probability.ipynb**](ml-103/math_006_05_law_of_total_probability.ipynb) - Law of total probability
- [**math_006_06_bayes_theorem.ipynb**](ml-103/math_006_06_bayes_theorem.ipynb) - Bayes' theorem and applications
- [**math_006_07_independence.ipynb**](ml-103/math_006_07_independence.ipynb) - Independence and conditional independence
- [**math_006_08_probability_of_and.ipynb**](ml-103/math_006_08_probability_of_and.ipynb) - AND operations in probability
- [**math_006_09_log_probabilities.ipynb**](ml-103/math_006_09_log_probabilities.ipynb) - Logarithmic probabilities
- [**math_006_10_many_flips.ipynb**](ml-103/math_006_10_many_flips.ipynb) - Repeated trials and distributions
- [**math_006_11_core_probability_examples.ipynb**](ml-103/math_006_11_core_probability_examples.ipynb) - Core probability examples
- [**math_006_12_probability_review.ipynb**](ml-103/math_006_12_probability_review.ipynb) - Comprehensive probability review

### Random Variables

- [**math_007_01_random_variables.ipynb**](ml-103/math_007_01_random_variables.ipynb) - Introduction to random variables

---

## 📚 ML-201: Data Analysis & Feature Engineering

The `ml-201/` directory contains practical data science and exploratory data analysis (EDA) techniques:

### Data Cleaning & Preprocessing

- [**eda_001_01_handling_missing_values.ipynb**](ml-201/eda_001_01_handling_missing_values.ipynb) - Handling missing data, covering:
  - Missing value detection and visualization
  - Imputation strategies (mean, median, forward-fill, etc.)
  - Impact of missing data on analysis
  - Best practices for different data types

- [**eda_001_02_scaling_and_normalization.ipynb**](ml-201/eda_001_02_scaling_and_normalization.ipynb) - Feature scaling techniques, covering:
  - Standardization and normalization
  - Min-Max scaling and Z-score normalization
  - When to apply each technique
  - Impact on model performance

- [**eda_001_03_parsing_dates.ipynb**](ml-201/eda_001_03_parsing_dates.ipynb) - Date and time handling, covering:
  - Parsing and converting date formats
  - Time series data preparation
  - Extracting temporal features
  - Handling timezones and ambiguities

- [**eda_001_04_character_encodings.ipynb**](ml-201/eda_001_04_character_encodings.ipynb) - Character encoding techniques, covering:
  - Understanding different encodings
  - Detecting and fixing encoding issues
  - Unicode and multilingual data handling
  - Text preprocessing

- [**eda_001_05_inconsistent_data_entry.ipynb**](ml-201/eda_001_05_inconsistent_data_entry.ipynb) - Data consistency and cleaning, covering:
  - Identifying inconsistencies and duplicates
  - Standardizing entries
  - Handling typos and variations
  - Data validation techniques

- [**eda_001_06_other_techniques.ipynb**](ml-201/eda_001_06_other_techniques.ipynb) - Additional cleaning techniques

### Real-World Data Analysis

- [**eda_002_01_nfl_play_by_play_2009.ipynb**](ml-201/eda_002_01_nfl_play_by_play_2009.ipynb) - NFL play-by-play data analysis, covering:
  - Large dataset exploration and handling
  - Domain-specific data understanding
  - Statistical analysis of sports data
  - Visualization of temporal patterns

### Feature Engineering

- [**eda_003_01_what_is_feature_engineering.ipynb**](ml-201/eda_003_01_what_is_feature_engineering.ipynb) - Introduction to feature engineering, covering:
  - Feature engineering principles and importance
  - Domain knowledge and feature creation
  - Feature validation and selection
  - Pipeline development

- [**eda_003_02_mutual_information.ipynb**](ml-201/eda_003_02_mutual_information.ipynb) - Mutual information for feature selection, covering:
  - Information theory basics
  - Mutual information calculation
  - Feature relevance scoring
  - Handling redundant features

- [**eda_003_03_creating_features.ipynb**](ml-201/eda_003_03_creating_features.ipynb) - Feature creation techniques, covering:
  - Polynomial features
  - Interaction terms
  - Binning and discretization
  - Domain-specific feature derivation

- [**eda_003_04_clustering_with_kmeans.ipynb**](ml-201/eda_003_04_clustering_with_kmeans.ipynb) - K-means clustering for feature creation, covering:
  - Clustering algorithms and applications
  - K-means implementation and tuning
  - Distance metrics and similarity
  - Cluster-based features

- [**eda_003_05_principal_component_analysis.ipynb**](ml-201/eda_003_05_principal_component_analysis.ipynb) - PCA for dimensionality reduction, covering:
  - Principal component analysis theory
  - Variance explained and components
  - Feature extraction with PCA
  - Applications to high-dimensional data

- [**eda_003_06_target_encoding.ipynb**](ml-201/eda_003_06_target_encoding.ipynb) - Encoding categorical features with target information, covering:
  - Target encoding techniques
  - Handling categorical variables
  - Preventing data leakage
  - Comparison with other encoding methods

- [**eda_003_07_other_techniques.ipynb**](ml-201/eda_003_07_other_techniques.ipynb) - Advanced feature engineering techniques

### Exploratory Data Analysis

- [**eda_004_01_univariate_analysis.ipynb**](ml-201/eda_004_01_univariate_analysis.ipynb) - Single variable analysis, covering:
  - Distribution analysis and summary statistics
  - Outlier detection
  - Histogram and density plots
  - Understanding feature distributions

- [**eda_004_02_bivariate_analysis.ipynb**](ml-201/eda_004_02_bivariate_analysis.ipynb) - Two-variable relationships, covering:
  - Correlation and association
  - Scatter plots and trend analysis
  - Categorical relationships
  - Covariance and dependency

- [**eda_004_03_multivariate_analysis.ipynb**](ml-201/eda_004_03_multivariate_analysis.ipynb) - Multi-variable analysis, covering:
  - Interactions between multiple features
  - High-dimensional data visualization
  - Clustering and segmentation
  - Principal component analysis

- [**eda_004_04_descriptive_statistics.ipynb**](ml-201/eda_004_04_descriptive_statistics.ipynb) - Summary statistics and distributions, covering:
  - Central tendency (mean, median, mode)
  - Dispersion (variance, std dev, IQR)
  - Skewness and kurtosis
  - Distribution identification

- [**eda_004_05_correlation_and_relationships.ipynb**](ml-201/eda_004_05_correlation_and_relationships.ipynb) - Correlation analysis, covering:
  - Pearson, Spearman, Kendall correlations
  - Correlation matrices and heatmaps
  - Spurious correlations
  - Causation vs. correlation

- [**eda_004_06_time_series_exploration.ipynb**](ml-201/eda_004_06_time_series_exploration.ipynb) - Time series analysis, covering:
  - Temporal patterns and trends
  - Seasonality and stationarity
  - Autocorrelation and lags
  - Forecasting foundations

- [**eda_004_07_geospatial_analysis.ipynb**](ml-201/eda_004_07_geospatial_analysis.ipynb) - Geographic data analysis, covering:
  - Coordinate systems and mapping
  - Spatial clustering
  - Distance and proximity analysis
  - Visualization techniques

---

## 📚 API Guides & References

Comprehensive API reference guides for popular machine learning libraries:

### [api_guides/](api_guides/)

- [**torch_api_guide.ipynb**](api_guides/torch_api_guide.ipynb) - PyTorch Complete Reference (20 sections)
  - Tensor creation and manipulation
  - Neural network modules
  - Automatic differentiation and autograd
  - Optimizers and learning rate schedulers
  - Loss functions and activation functions
  - Device management (CPU/GPU)
  - **Advanced topics**: C++ integration, calling conventions, pybind11, .pyi stub files
  - Complete training examples

- [**numpy_api_guide.ipynb**](api_guides/numpy_api_guide.ipynb) - NumPy Complete Reference (21 sections)
  - Array creation and properties
  - Mathematical operations and ufuncs
  - Linear algebra and decompositions
  - Reduction operations and broadcasting
  - Indexing and slicing
  - Sorting and searching
  - Random number generation
  - **Advanced topics**: C internals, calling conventions, ctypes integration, .pyi stub files, memory layout, strides
  - Function overloading and type dispatch
  - Complete ML data pipeline example

---

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start Jupyter:
   ```bash
   jupyter notebook
   ```

4. Choose your learning path:
   - **Beginners**: Start with `ml-101/math_001.ipynb`
   - **Math-focused**: Work through ml-101 → ml-102 → ml-103
   - **Data-focused**: Jump to `ml-201/` for practical data analysis
   - **API Reference**: Open `api_guides/numpy_api_guide.ipynb` or `api_guides/torch_api_guide.ipynb`

## Learning Path Recommendations

### 🎓 Complete Curriculum (Recommended for comprehensive understanding)

1. **Start with ML-101** (Linear Algebra Fundamentals - 12 notebooks)
   - Learn the mathematical foundations of machine learning
   - Master matrices, vectors, transformations, eigenvalues, and PCA
   - Build intuition about data structures and operations in ML

2. **Progress to ML-102** (Calculus & Optimization - 12 notebooks)
   - Understand derivatives from symbolic, numerical, and automatic approaches
   - Master gradient descent and optimization techniques
   - Learn loss functions and neural network fundamentals
   - Compare 4 different differentiation approaches (SymPy, NumPy, JAX, PyTorch)

3. **Continue with ML-103** (Combinatorics & Probability - 12+ notebooks)
   - Build strong intuition for counting principles and distributions
   - Master permutations, combinations, and probability theory
   - Understand Bayes' theorem and conditional probability
   - Essential for statistical inference and machine learning

4. **Apply to ML-201** (Data Analysis & Features - 20+ notebooks)
   - Learn practical data cleaning and preprocessing
   - Master feature engineering and dimensionality reduction
   - Perform real-world exploratory data analysis
   - Work with actual datasets (NFL, Kaggle competitions)

### 🚀 Fast Track (Practical data science)
- Skip to **ML-201** for immediate practical skills
- Refer back to **ml-101** and **ml-102** as needed for mathematical foundations

### 💡 Deep Learning Track
- **ML-101** → **ML-102** → API Guides (PyTorch) for neural network development
- Focus on gradient descent, automatic differentiation, and neural networks

### 📊 Data Science Track
- **ML-101** → **ML-201** → API Guides (NumPy) for data analysis
- Focus on arrays, linear algebra, and exploratory data analysis

### 📚 API Reference
- **NumPy API Guide**: Comprehensive reference for array operations, broadcasting, and internal mechanisms
- **PyTorch API Guide**: Complete guide to tensors, neural networks, and GPU computing

Each notebook includes:
- ✅ Interactive visualizations
- ✅ Executable Python code examples
- ✅ Real-world applications
- ✅ Practice exercises
- ✅ Performance tips and best practices

## License

See [LICENSE](LICENSE) for details.
