include("../src/CenteredSparseMatrix.jl")
#include("sparse_centered.jl")

using Main.CenteredSparseMatrix
using Test



# Just some setup

n = 10
m = 3
d = 0.4

k = 2

#for T = Int64
#for T in (Int32, Int64, Float32, Float64, Complex64, Complex128)
for T in (Int32, Float32, Float64, Complex64, Complex128)
    print(T, " started... ")
    A = sprand(T, n, m, d);
    Acd = full(A .- mean(A, 1));
    Acs = CenteredSparseCSC(A);

    # test constructors
    Acs2 = CenteredSparseCSC(full(A));
    Acs3 = CenteredSparseCSC(
             A.rowval,
             vcat([repeat([j], inner=A.colptr[j+1] - A.colptr[j])
                     for j in 1:length(A.colptr) - 1]...),
             A.nzval, A.m, A.n)
    @test isapprox(Acs, Acs2)
    @test isapprox(Acs, Acs3)

    outT = typeof(zero(T) / one(T))

    x = rand(T, m);
    X = rand(T, m, k);
    y = rand(T, n);
    Y = rand(T, n, k);

    z = zeros(outT, size(A, 1));
    Z = zeros(outT, size(A, 1), size(X, 2));
    w = zeros(outT, size(A, 2));
    W = zeros(outT, size(A, 2), size(X, 2));


    # the actual tests

    @test isapprox(Acd * x, Acs * x)
    @test isapprox(Acd * x, A_mul_B!(z, Acs, x))

    @test isapprox(Acd * X, Acs * X)
    @test isapprox(Acd * X, A_mul_B!(Z, Acs, X))
    @test isapprox(Acd * X, A_mul_B!(Z, Acs, X)) #yes twice, zero Z test

    @test isapprox(Acd' * y, Ac_mul_B(Acs, y))
    @test isapprox(Acd' * y, Ac_mul_B!(w, Acs, y))
    @test isapprox(Acd' * y, Acs' * y)
    @test isapprox(Acd' * y, Acs'y)

    @test isapprox(Acd.' * y, At_mul_B(Acs, y))
    @test isapprox(Acd.' * y, At_mul_B!(w, Acs, y))
    @test isapprox(Acd.' * y, Acs.' * y)
    @test isapprox(Acd.' * y, Acs.'y)

    @test isapprox(Acd' * Y, Ac_mul_B(Acs, Y))
    @test isapprox(Acd' * Y, Ac_mul_B!(W, Acs, Y))
    @test isapprox(Acd' * Y, Acs' * Y)
    @test isapprox(Acd' * Y, Acs'Y)

    @test isapprox(Acd.' * Y, At_mul_B!(W, Acs, Y))
    @test isapprox(Acd.' * Y, At_mul_B(Acs, Y))
    @test isapprox(Acd.' * Y, Acs.' * Y)
    @test isapprox(Acd.' * Y, Acs.'Y)

    if !(T <: Complex)
        Acd2 = copy(Acd)
        scale_sd!(Acs2)
        scale!(Acd2, 1 ./ vec(std(Acd2, 1)))
        # can't find an appropriate scale_sd...?!
        #@test isapprox(Acd2, scale_sd(Acs2))
        @test isapprox(Acd2, Acs2)
        # produces floats, so can't do in place, should do for scale_sd()
        if !(T <: Integer)
            Ad = full(A)
            As = copy(A)
            scale!(Ad, 1 ./ vec(std(Ad, 1)))
            # can't find an appropriate scale_sd...?!
            #@test isapprox(Ad, scale_sd(As))
            scale_sd!(As)
            @test isapprox(Ad, As)
        end
    end

    println(T, " complete")
end

