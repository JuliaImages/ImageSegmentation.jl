@testset "Felzenszwalb" begin

    img = zeros(Gray{N0f8}, 10, 10)
    img[2:3, 2:3] = 0.2
    img[2, 8] = 0.5
    img[5:6, 5:6] = 0.8
    T = Gray{N0f8}

    result = felzenszwalb(img, 1)

    @test length(result.segment_labels) == 4
    @test result.segment_labels == collect(1:4)

    @test all(label->(label==result.image_indexmap[2,2]), result.image_indexmap[2:3, 2:3])
    @test all(label->(label==result.image_indexmap[5,5]), result.image_indexmap[5:6, 5:6])

    @test result.segment_means[result.image_indexmap[1,1]] == zero(Images.accum(T))
    @test result.segment_means[result.image_indexmap[2,2]] == Images.accum(T)(img[2,2])
    @test result.segment_means[result.image_indexmap[2,8]] == Images.accum(T)(img[2,8])
    @test result.segment_means[result.image_indexmap[5,5]] == Images.accum(T)(img[5,5])

    @test result.segment_pixel_count[result.image_indexmap[1,1]] == 91
    @test result.segment_pixel_count[result.image_indexmap[2,2]] == 4
    @test result.segment_pixel_count[result.image_indexmap[2,8]] == 1
    @test result.segment_pixel_count[result.image_indexmap[5,5]] == 4


    img = zeros(RGB{Float64}, 10, 10)
    img[2:3, 2:3] = RGB(0.2,0.2,0.2)
    img[2, 8] = RGB(0.5,0.5,0.5)
    img[5:6, 5:6] = RGB(0.8,0.8,0.8)
    T = RGB{Float64}

    result = felzenszwalb(img, 1, 2)

    @test length(result.segment_labels) == 3
    @test result.segment_labels == collect(1:3)

    @test result.image_indexmap[1, 1] == result.image_indexmap[2, 8]
    @test all(label->(label==result.image_indexmap[2,2]), result.image_indexmap[2:3, 2:3])
    @test all(label->(label==result.image_indexmap[5,5]), result.image_indexmap[5:6, 5:6])

    @test result.segment_means[result.image_indexmap[1,1]] ≈ RGB{Float64}(0.5/92, 0.5/92, 0.5/92)
    @test result.segment_means[result.image_indexmap[2,2]] == Images.accum(T)(img[2,2])
    @test result.segment_means[result.image_indexmap[5,5]] == Images.accum(T)(img[5,5])

    @test result.segment_pixel_count[result.image_indexmap[1,1]] == 92
    @test result.segment_pixel_count[result.image_indexmap[2,2]] == 4
    @test result.segment_pixel_count[result.image_indexmap[5,5]] == 4

    # issue 20
    img = falses(10, 10)
    img[2:3, 2:3] = true
    img[2, 8] = true
    img[5:6, 5:6] = true
    T = Bool

    result = felzenszwalb(img, 1, 2)

    @test length(result.segment_labels) == 3
    @test result.segment_labels == collect(1:3)

    @test result.image_indexmap[1, 1] == result.image_indexmap[2, 8]
    @test all(label->(label==result.image_indexmap[2,2]), result.image_indexmap[2:3, 2:3])
    @test all(label->(label==result.image_indexmap[5,5]), result.image_indexmap[5:6, 5:6])

    @test result.segment_means[result.image_indexmap[1,1]] ≈ 1/92
    @test result.segment_means[result.image_indexmap[2,2]] == Images.accum(T)(img[2,2])
    @test result.segment_means[result.image_indexmap[5,5]] == Images.accum(T)(img[5,5])

    @test result.segment_pixel_count[result.image_indexmap[1,1]] == 92
    @test result.segment_pixel_count[result.image_indexmap[2,2]] == 4
    @test result.segment_pixel_count[result.image_indexmap[5,5]] == 4
end
