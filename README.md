# Learn ML

A comprehensive learning resource for machine learning fundamentals, built on mathematical foundations.

## About

This repository contains interactive Jupyter notebooks covering essential mathematics concepts and ML algorithms, progressing from basic linear algebra to principal component analysis (PCA).

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

We recommend following the notebooks in numerical order (math_001 → math_012) to build a solid understanding of the mathematical foundations needed for machine learning.

## License

See [LICENSE](LICENSE) for details.
