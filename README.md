# Learn ML

A comprehensive learning resource for machine learning fundamentals, built on mathematical foundations.

## About

This repository contains interactive Jupyter notebooks covering essential mathematics concepts and ML algorithms, progressing from basic linear algebra through calculus and automatic differentiation to combinatorics and probability.

## 📚 ML-101 Curriculum

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

## 📚 ML-102 Curriculum

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

## 📚 ML-103 Curriculum

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

4. Navigate to `ml-101/` and open any notebook to get started!

## Learning Path

### Recommended Sequence

1. **Start with ML-101** (Linear Algebra Fundamentals)
   - Begin with math_001 and progress through all 12 notebooks
   - Covers arrays, matrices, transformations, eigenvalues, and PCA
   - Foundation for understanding data structures in ML

2. **Continue with ML-102** (Calculus & Neural Networks)
   - Master derivatives, automatic differentiation, and optimization
   - Understand loss functions and gradient descent
   - Learn the basics of perceptrons and neural network fundamentals
   - Compare symbolic, numerical, and automatic differentiation approaches

3. **Conclude with ML-103** (Combinatorics & Probability)
   - Build intuition for counting, distributions, and discrete mathematics
   - Master permutations, combinations, and stars-and-bars method
   - Understand constraints and transformations in counting problems
   - Essential background for probability theory and statistical inference

Each notebook builds on previous concepts, with progressive difficulty and increasingly practical applications. Interactive visualizations help solidify understanding of abstract mathematical concepts.

## License

See [LICENSE](LICENSE) for details.
