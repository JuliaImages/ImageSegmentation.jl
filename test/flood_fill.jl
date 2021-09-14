using ImageSegmentation
using ImageSegmentation.Colors
using ImageSegmentation.FixedPointNumbers
using FileIO
using Statistics
using SparseArrays
using Test

@testset "flood_fill" begin
    # 0d
    a = reshape([true])
    @test flood(identity, a, CartesianIndex()) == a
    @test_throws ArgumentError flood(!, a, CartesianIndex())
    # 1d
    a = 1:7
    @test flood(==(2), a, CartesianIndex(2)) == (a .== 2)
    @test_throws ArgumentError flood(==(2), a, CartesianIndex(3))
    @test flood(x -> 1 < x < 4, a, CartesianIndex(2)) == [false, true, true, false, false, false, false]
    @test flood(isinteger, a, CartesianIndex(2)) == trues(7)
    # 2d
    ab = [true false false false;
         true true false false;
         true false false true;
         true true true true]
    an0f8 = N0f8.(ab)
    agray = Gray.(an0f8)
        for (f, a) in ((identity, ab), (==(1), an0f8), (==(1), agray))
        for idx in CartesianIndices(a)
            if f(a[idx])
                @test flood(f, a, idx) == a
            else
                @test_throws ArgumentError flood(f, a, idx)
            end
        end
    end
    @test flood(identity, ab, Int16(1)) == ab
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
            @test flood(identity, a, idx) == a
        else
            @test_throws ArgumentError flood(identity, a, idx)
        end
    end
    # Colors
    path = download("https://github.com/JuliaImages/juliaimages.github.io/raw/source/docs/src/pkgs/segmentation/assets/flower.jpg")
    img = load(path)
    seg = flood(img, CartesianIndex(87,280); thresh=0.3*sqrt(3))   # TODO: eliminate the sqrt(3) when we transition to `abs2(c) = c ⋅ c`
    @test 0.2*length(seg) <= sum(seg) <= 0.25*length(seg)
    c = mean(img[seg])
    # N0f8 makes for easier approximate testing
    @test N0f8(red(c)) ≈ N0f8(0.855)
    @test N0f8(green(c)) ≈ N0f8(0.161)
    @test N0f8(blue(c)) ≈ N0f8(0.439)

    # flood_fill!
    near3(x) = round(Int, x) == 3
    a0 = [range(2, 4, length=9);]
    a = copy(a0)
    idx = (length(a)+1)÷2
    dest = fill!(similar(a, Bool), false)
    @test flood_fill!(near3, dest, a, idx) == (round.(a) .== 3)
    a = copy(a0)
    flood_fill!(near3, a, idx; fillvalue=-1)
    @test a == [near3(a0[i]) ? -1 : a[i] for i in eachindex(a)]
    a = copy(a0)
    @test_throws ArgumentError flood_fill!(near3, a, idx; fillvalue=-1, isfilled=near3)
    # warning
    a = [1:7;]
    @test_logs (:warn, r"distinct.*incomplete") flood_fill!(<(5), a, 1; fillvalue=3)
    @test a == [3,3,3,4,5,6,7]
    a = [1:7;]
    dest = fill(-1, size(a))
    @test_logs flood_fill!(<(5), dest, a, 1; fillvalue=3)   # no warnings
    @test dest == [3,3,3,3,-1,-1,-1]
    a = [1:7;]
    @test_logs flood_fill!(<(5), a, 1; fillvalue=11)
    @test a == [11,11,11,11,5,6,7]

    # This mimics a "big data" application in which we have several structures we want
    # to label with different segment numbers, and the `src` array is too big to fit
    # in memory.
    # It would be better to use a package like SparseArrayKit, which allows efficient
    # insertions and supports arbitrary dimensions.
    a = Bool[0 0 0 0 0 0 1 1;
             1 1 0 0 0 0 0 0]
    dest = spzeros(Int, size(a)...)   # stores the nonzero indexes in a Dict
    flood_fill!(identity, dest, a, CartesianIndex(2, 1); fillvalue=1)
    flood_fill!(identity, dest, a, CartesianIndex(1, 7); fillvalue=2)
    @test dest == [0 0 0 0 0 0 2 2;
                   1 1 0 0 0 0 0 0]
end
