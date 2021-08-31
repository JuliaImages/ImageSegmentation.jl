@testset "Clustering" begin
    # Test the example in the docs
    path = download("https://github.com/JuliaImages/juliaimages.github.io/raw/source/docs/src/pkgs/segmentation/assets/flower.jpg")
    img = load(path)
    r = fuzzy_cmeans(img, 3, 2)
    @test size(r.centers) == (3,3)
    @test size(r.weights, 1) == length(img)
    @test all(≈(1), sum(r.weights; dims=2))
    cmin, cmax = extrema(sum(r.weights; dims=1))
    @test cmax < 3*cmin
    # Make sure we support OffsetArrays
    imgo = OffsetArray(img, (-1, -1))
    ro = fuzzy_cmeans(imgo, 3, 2)
    @test size(ro.centers) == (3,3)

    # Also with kmeans
    r = kmeans(img, 3)
    nc = last.(sort([pr for pr in segment_pixel_count(r)]; by=first))
    @test sum(nc) == length(img)
    @test 3*minimum(nc) > maximum(nc)
    ro = kmeans(imgo, 3)
    nc = last.(sort([pr for pr in segment_pixel_count(ro)]; by=first))
    @test sum(nc) == length(img)
end
