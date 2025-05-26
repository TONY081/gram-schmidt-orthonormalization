# gram-schmidt-orthonormalization
Python implementation of the Gram-Schmidt process to compute orthonormal bases and the dimension of vector spans.
# Gram-Schmidt Orthonormalization

This repository contains Python implementations of the Gram-Schmidt process for generating an orthonormal basis from a set of vectors.

## Features

- `gsBasis4(A)`: Applies the Gram-Schmidt process to 4 vectors (columns of matrix A).
- `gsBasis(A)`: General implementation for any number of vectors.
- `dimensions(A)`: Calculates the dimension of the space spanned by the input vectors.

The code uses NumPy for efficient numerical computations and handles linear dependence by zeroing out near-zero vectors (tolerance set to 1e-14).

## Usage

Simply import the functions into your Python environment or script:

```python
from gram_schmidt import gsBasis, gsBasis4, dimensions
