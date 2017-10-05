# CenteredSparseMatrix

This package provides the `CenteredSparseCSC` type. It assumes that you want to
re-center a sparse matrix based on the mean of each column. However, you can
supply a custom centered value for each column. There are no methods for other
forms of centering (e.g. the rows of the matrix). Currently only sparse-dense
multiply and conjugate-multiply are implemented, with just enough indexing to
make the display of the matrix work in a reasonable manner. That said, using
the type is easy:

```
#Pkg.clone("git@github.com:jsams/CenteredSparseMatrix.git")

using CenteredSparseMatrix

X = sprand(10, 3, 0.6)
X_cent_sparse = CenteredSparseCSC(X)
X_cent_dense = full(X) .- mean(X, 1)

y = rand(3)
Y = rand(3, 5)
z = rand(10)
Z = rand(10, 4)

isapprox(X_cent_sparse * y, X_cent_dense * y)
isapprox(X_cent_sparse * Y, X_cent_dense * Y)
isapprox(X_cent_sparse' * z, X_cent_dense' * z)
isapprox(X_cent_sparse' * Z, X_cent_dense' * Z)
```

The key point is that the sparsity structure of the matrix is left unchanged,
the centering of the zero-elements is done on-demand, and where possible, algorithms
take advantage of knowing the column-constant mean value.

For the matrix multiplications `X_cent_sparse  * Y` and `X_cent_sparse' * Z`, there
is nearly zero overhead compared to the plain sparse multiplications `X * Y`
and `X' * Z`.
