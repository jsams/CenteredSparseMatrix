# CenteredSparseMatrix

This package provides the `CenteredSparseCSC` type. It assumes that you want to
re-center a sparse matrix based on the mean of each column. However, you can
supply a custom centered value for each column. There are no methods for other
forms of centering (e.g. the rows of the matrix). Currently only sparse-dense
multiply and conjugate-multiply are implemented, with just enough indexing to
make the display of the matrix work in a reasonable manner. That said, using
the type is easy:

```
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

I wrote this to enable computing things like a centered partial SVD on very
large (~3 billion non-zero elements, in a 27-trillion cell matrix) without
having to actually store the 27 trillion cells (that would be 216 TiB of data
as opposed to 24 GiB)

I've tried adapting the gapxy algorithms in an efficient manner to this use
case, but suggestions to improve efficiency are welcome.

