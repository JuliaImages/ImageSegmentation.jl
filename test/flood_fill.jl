using ImageSegmentation
using ImageSegmentation.Colors
using ImageSegmentation.FixedPointNumbers
using FileIO
using Statistics
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
    ab = [true false false false;
         true true false false;
         true false false true;
         true true true true]
    an0f8 = N0f8.(ab)
    agray = Gray.(an0f8)
        for (f, a) in ((identity, ab), (==(1), an0f8), (==(1), agray))
        for idx in CartesianIndices(a)
            if f(a[idx])
                @test flood_fill(f, a, idx) == a
            else
                @test_throws ArgumentError flood_fill(f, a, idx)
            end
        end
    end
    @test flood_fill(identity, ab, Int16(1)) == ab
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
    # Colors
    path = download("https://github.com/JuliaImages/juliaimages.github.io/raw/source/docs/src/pkgs/segmentation/assets/flower.jpg")
    img = load(path)
    seg = flood_fill(img, CartesianIndex(87,280); thresh=0.3)
    @test 0.2*length(seg) <= sum(seg) <= 0.25*length(seg)
    c = mean(img[seg])
    # N0f8 makes for easier approximate testing
    @test N0f8(red(c)) ≈ N0f8(0.855)
    @test N0f8(green(c)) ≈ N0f8(0.161)
    @test N0f8(blue(c)) ≈ N0f8(0.439)
end
