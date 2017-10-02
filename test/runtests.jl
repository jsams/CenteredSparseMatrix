using CenteredSparseMatrix
using Base.Test

# write your own tests here
@test 1 == 2
include("sparse_centered.jl")
include("sparse_centered_ref.jl")

import Main.CenteredSparseRef
using Main.CenteredSparse

n = 100
m = 50
d = 0.3

k = 30

A = sprand(n, m, d);
Acd = full(A .- mean(A, 1));
Acs = CenteredSparseCSC(A);
Acr = CenteredSparseRef.CenteredSparseCSC(A);


x = rand(m);
X = rand(m, k);
y = rand(n);
Y = rand(n, k);


# accuracy tests

#####################
# fast implementation
#####################

isapprox(Acd * x, Acs * x)
isapprox(Acd * X, Acs * X)
isapprox(Acd' * y, Ac_mul_B(Acs, y))
isapprox(Acd' * y, Acs' * y)
isapprox(Acd' * y, Acs'y)
isapprox(Acd' * Y, Ac_mul_B(Acs, Y))
isapprox(Acd' * Y, Acs' * Y)
isapprox(Acd' * Y, Acs'Y)


##########################
# reference implementation (works)
##########################
isapprox(Acd * x, Acr * x)
isapprox(Acd * X, Acr * X)
isapprox(Acd' * y, Ac_mul_B(Acr, y))
isapprox(Acd' * y, Acr' * y)
isapprox(Acd' * Y, Ac_mul_B(Acr, Y))
isapprox(Acd' * Y, Acr' * Y)






##########################
# benchmarks and profiling
##########################

@time z = A * x;
@time z = Acs * x;
@time z = Acr * x;
Profile.clear()
@profile z = Acs * x;
Profile.print()

@time Z = A * X;
@time Z = Acs * X;
@time Z = Acr * X;
Profile.clear()
Profile.clear_malloc_data()
@profile Z = Acs * X;
Profile.print()

@time w = A' * y;
@time w = Acs' * y;
@time w = Acr' * y;
Profile.clear()
@profile w = Acs' * y;
Profile.print()

@time W = A' * Y;
@time W = Acs' * Y;
@time W = Acr' * Y;
Profile.clear()
@profile W = Acs' * Y;
Profile.print()



##################################################################
# SVD Tests (test in place multiplies and putting it all together)
##################################################################

using RDatasets
X = convert(Array{Float64, 2}, Array(dataset("datasets", "mtcars")[:, 2:8]))

DENS = 0.3
k = 50
X = rand(4000, 300);
X0 = copy(X);
tozeroidx = rand(1:length(X), convert(Int64, floor(length(X) * DENS)));
X0[tozeroidx] = 0;

X0cd = (X0 .- mean(X0, 1));
X0us = sparse(X0)
X0cs = CenteredSparseCSC(X0);
X0cr = CenteredSparseRef.CenteredSparseCSC(X0cs.A);

@time autod = svds(X0cd, nsv=k);
@time autou = svds(X0us, nsv=k);
@time autos = svds(X0cs, nsv=k);
@time autor = svds(X0cr, nsv=k);

isapprox(autod[1], autos[1])
isapprox(autod[1], autor[1])


# instrumental accuracy
n = 10
N = collect('A':'J')
to_remove = []
to_remove = [1]
to_remove = [n]
to_remove = [7]
to_remove = [1 2 3]
to_remove = [1 4 n]
to_remove = [2 4 n]
to_remove = [2 n-1 n]
to_remove = [2 8 n-1]
to_remove = collect(1:n)

to_keep = zeros(Int64, n)
idx = inv_index!(view(to_keep, 1:(n - length(to_remove))), to_remove, n)
N[idx]

# would be good to check that this is actually better than setdiff



# debug internal functions more easily
# A = Acs
# x = X
# y = zeros(size(A, 1), size(x, 2))
# n, m = size(A)
# idx_pool = zeros(Int64, n)


