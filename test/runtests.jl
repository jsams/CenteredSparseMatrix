include("../src/CenteredSparseMatrix.jl")
#include("sparse_centered.jl")

using Main.CenteredSparseMatrix
using Base.Test



# Just some setup

n = 50
m = 30
d = 0.6

k = 2

A = sprand(n, m, d);
Acd = full(A .- mean(A, 1));
Acs = CenteredSparseCSC(A);

x = rand(m);
X = rand(m, k);
y = rand(n);
Y = rand(n, k);

z = zeros(size(A, 1));
Z = zeros(size(A, 1), size(X, 2));
w = zeros(size(A, 2))
W = zeros(size(A, 2), size(X, 2))


# the actual tests

@test isapprox(Acd * x, Acs * x)
@test isapprox(Acd * x, A_mul_B!(z, Acs, x))

@test isapprox(Acd * X, Acs * X)
@test isapprox(Acd * X, A_mul_B!(Z, Acs, X))

@test isapprox(Acd' * y, Ac_mul_B(Acs, y))
@test isapprox(Acd' * y, Acs' * y)
@test isapprox(Acd' * y, Acs'y)

@test isapprox(Acd' * Y, Ac_mul_B(Acs, Y))

@test isapprox(Acd' * Y, Acs' * Y)
@test isapprox(Acd' * Y, Acs'Y)

