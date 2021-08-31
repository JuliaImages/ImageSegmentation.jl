using ImageSegmentation
using Test

@testset "flood_fill" begin
    # 0d
    a = reshape([true])
    @test flood_fill(identity, a, CartesianIndex()) == a
    @test_throws ArgumentError flood_fill(!, a, CartesianIndex())
    # 1d
    a = 1:7
    @test flood_fill(==(2), a, CartesianIndex(2)) == (a .== 2)
    @test_throws ArgumentError flood_fill(==(2), a, CartesianIndex(3))
    @test flood_fill(x -> 1 < x < 4, a, CartesianIndex(2)) == [false, true, true, false, false, false, false]
    @test flood_fill(isinteger, a, CartesianIndex(2)) == trues(7)
    # 2d
    a = [true false false false;
         true true false false;
         true false false true;
         true true true true]
    for idx in CartesianIndices(a)
        if a[idx]
            @test flood_fill(identity, a, idx) == a
        else
            @test_throws ArgumentError flood_fill(identity, a, idx)
        end
    end
    # 3d
    k = 10
    a = falses(k, k, k)
    idx = CartesianIndex(1,1,1)
    incs = [CartesianIndex(1,0,0), CartesianIndex(0,1,0), CartesianIndex(0,0,1)]
    a[idx] = true
    while any(<(k), Tuple(idx))
        d = rand(1:3)
        idx += incs[d]
        idx = min(idx, CartesianIndex(k,k,k))
        a[idx] = true
    end
    for idx in eachindex(a)
        if a[idx]
            @test flood_fill(identity, a, idx) == a
        else
            @test_throws ArgumentError flood_fill(identity, a, idx)
        end
    end
end
