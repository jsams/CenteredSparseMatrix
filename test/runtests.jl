include("../src/CenteredSparseMatrix.jl")
#include("sparse_centered.jl")

using Main.CenteredSparseMatrix
using Base.Test



# Just some setup

n = 100
m = 50
d = 0.3

k = 30

A = sprand(n, m, d);
Acd = full(A .- mean(A, 1));
Acs = CenteredSparseCSC(A);


x = rand(m);
X = rand(m, k);
y = rand(n);
Y = rand(n, k);

@test isapprox(Acd * x, Acs * x)
@test isapprox(Acd * X, Acs * X)
@test isapprox(Acd' * y, Ac_mul_B(Acs, y))
@test isapprox(Acd' * y, Acs' * y)
@test isapprox(Acd' * y, Acs'y)
@test isapprox(Acd' * Y, Ac_mul_B(Acs, Y))
@test isapprox(Acd' * Y, Acs' * Y)
@test isapprox(Acd' * Y, Acs'Y)


