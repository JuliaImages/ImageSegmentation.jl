using ImageFiltering

@testset "watershed" begin

    img = zeros(100, 100)
    img[25:75, 25:75] .= 1
    img[26:74, 26:74] .= 0

    markers = zeros(Int, size(img))
    markers[1, 1] = 1
    markers[50, 50] = 2

    @testset "classic watershed" begin
        # with classic watershed, we expect the top left label should spread around
        # the center label.
        result = watershed(img, markers)

        @test length(result.segment_labels) == 2
        @test result.segment_labels == collect(1:2)
        @test all(label->(label==result.image_indexmap[50,50]), result.image_indexmap[26:74,26:74])
    end

    @testset "masked watershed" begin
        mask = trues(size(img))
        mask[60:70, 60:70] .= false

        result = watershed(img, markers, mask=mask)
        labels = labels_map(result)

        # where the mask is false, no label should be assigned
        @test sum(labels[.~ mask]) == 0

        result = watershed(img, markers, compactness=10.0, mask=mask)
        labels = labels_map(result)

        # where the mask is false, no label should be assigned
        @test sum(labels[.~ mask]) == 0
    end

    @testset "compact watershed" begin
        result = watershed(img, markers, compactness=10.0)
        labels = labels_map(result)

        # since this is using the compact algorithm with a high value for
        # compactness, the boundary between labels 1 and 2 should occur halfway
        # between the two markers
        @test sum(labels .== 1) == sum(1:50)
    end

    @testset "h-minima transform" begin
        img = ones(15, 15)
        #minima of depth 0.2
        img[3:5, 3:5] .= 0.9
        img[4,4] = 0.8
        #minima of depth 0.7
        img[9:13, 9:13] .= 0.8
        img[10:12, 10:12] .= 0.7
        img[11,11] = 0.3

        out = hmin_transform(img, 0.25)

        @test findlocalminima(img) == [CartesianIndex(4, 4), CartesianIndex(11, 11)]
        @test findlocalminima(out) == [CartesianIndex(11, 11)]
    end
end
