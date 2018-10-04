@testset "MeanShift Segmentation" begin

    img = zeros(Gray{N0f8}, 10, 10)
    img[2:3, 2:3] .= 0.2
    img[2, 8] = 0.5
    img[5:6, 5:6] .= 0.8
    T = Gray{N0f8}
    TM = ImageSegmentation.meantype(T)

    result = meanshift(img, 8, 7/255)

    @test length(result.segment_labels) == 4
    @test result.segment_labels == collect(1:4)

    @test all(label->(label==result.image_indexmap[2,2]), result.image_indexmap[2:3, 2:3])
    @test all(label->(label==result.image_indexmap[5,5]), result.image_indexmap[5:6, 5:6])

    @test result.segment_means[result.image_indexmap[1,1]] == zero(TM)
    @test result.segment_means[result.image_indexmap[2,2]] == TM(img[2,2])
    @test result.segment_means[result.image_indexmap[2,8]] == TM(img[2,8])
    @test result.segment_means[result.image_indexmap[5,5]] == TM(img[5,5])

    @test result.segment_pixel_count[result.image_indexmap[1,1]] == 91
    @test result.segment_pixel_count[result.image_indexmap[2,2]] == 4
    @test result.segment_pixel_count[result.image_indexmap[2,8]] == 1
    @test result.segment_pixel_count[result.image_indexmap[5,5]] == 4
end
