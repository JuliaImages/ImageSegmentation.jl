using ImageFiltering

@testset "watershed" begin
    
    img = zeros(100, 100)
    img[25:75, 25:75] = 1

    img, _ = magnitude_phase(img)

    markers = zeros(Int, size(img))
    markers[1, 1] = 1
    markers[50, 50] = 2

    result = watershed(img, markers)

    @test length(result.segment_labels) == 2
    @test result.segment_labels == collect(1:2)
    @test all(label->(label==result.image_indexmap[50,50]), result.image_indexmap[26:74,26:74])


    img = ones(15, 15)
    #minima of depth 0.2
    img[3:5, 3:5] = 0.9
    img[4,4] = 0.8
    #minima of depth 0.7
    img[9:13, 9:13] = 0.8
    img[10:12, 10:12] = 0.7
    img[11,11] = 0.3

    out = hmin_transform(img, 0.25)

    @test findlocalminima(img) == [CartesianIndex(4, 4), CartesianIndex(11, 11)]
    @test findlocalminima(out) == [CartesianIndex(11, 11)] 
end