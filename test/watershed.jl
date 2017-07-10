using ImageFiltering

@testset "watershed" begin
    
    img = zeros(100, 100)
    img = zeros(Float64, (100,100))
    img[25:75, 25:75] = 1

    img, _ = magnitude_phase(img)

    markers = zeros(img)
    markers[1, 1] = 1
    markers[50, 50] = 2

    result = watershed(img, markers)

    @test length(result.segment_labels) == 2
    @test result.segment_labels == collect(1:2)
    @test all(label->(label==result.image_indexmap[50,50]), result.image_indexmap[26:74,26:74])

    # To do: Add more tests
end