of_mean_type(p::Colorant) = ImageSegmentation.meantype(typeof(p))(p)

@testset "Seeded Region Growing" begin
    # 2-D image
    img = zeros(Gray{N0f8}, 10, 10)
    img[6:10,4:8] .= 0.5
    img[3:7,2:6] .= 0.8
    seeds = [(CartesianIndex(3,9),1), (CartesianIndex(5,2),2), (CartesianIndex(9,7),3)]

    expected = ones(Int, 10, 10)
    expected[6:10,4:8] .= 3
    expected[3:7,2:6] .= 2
    expected_labels = [1,2,3]
    expected_means = Dict(1 => of_mean_type(img[3,9]), 2 => of_mean_type(img[5,2]), 3 => of_mean_type(img[9,7]))
    expected_count = Dict(1 => 56, 2 => 25, 3 => 19)

    seg = seeded_region_growing(img, seeds)
    @test all(label->(label in expected_labels), seg.segment_labels)
    @test all(label->(label in seg.segment_labels), expected_labels)
    @test expected_count == seg.segment_pixel_count
    @test expected_means == seg.segment_means
    @test seg.image_indexmap == expected

    # Custom neighbourhood using a function
    seg = seeded_region_growing(img, seeds, c->[CartesianIndex(c[1]-1,c[2]), CartesianIndex(c[1]+1,c[2]), CartesianIndex(c[1],c[2]-1), CartesianIndex(c[1],c[2]+1)])
    @test all(label->(label in expected_labels), seg.segment_labels)
    @test all(label->(label in seg.segment_labels), expected_labels)
    @test expected_count == seg.segment_pixel_count
    @test expected_means == seg.segment_means
    @test seg.image_indexmap == expected

    # Offset image
    img = centered(img)
    seeds = [(CartesianIndex(-2,4),1), (CartesianIndex(0,-3),2), (CartesianIndex(4,2),3)]
    expected = centered(expected)
    seg = seeded_region_growing(img, seeds)
    @test all(label->(label in expected_labels), seg.segment_labels)
    @test all(label->(label in seg.segment_labels), expected_labels)
    @test expected_count == seg.segment_pixel_count
    @test expected_means == seg.segment_means
    @test seg.image_indexmap == expected

    # Custom neighbourhood using a [3,3] vs [5,5] kernel
    img = zeros(Gray{N0f8}, 5, 5)
    img[2:4,2:4] .= 1
    img[3,3] = 0
    seeds = [(CartesianIndex(3,3),1), (CartesianIndex(2,3),2)]

    expected = fill(2,(5,5))
    expected[3,3] = 1
    expected_labels = [1,2]
    expected_means = Dict(1=>Gray{Float64}(0.0), 2=>Gray{Float64}(1/3))
    expected_count = Dict(1=>1, 2=>24)

    seg = seeded_region_growing(img, seeds, (3,3))
    @test all(label->(label in expected_labels), seg.segment_labels)
    @test all(label->(label in seg.segment_labels), expected_labels)
    @test expected_count == seg.segment_pixel_count
    @test all(label->(expected_means[label] ≈ seg.segment_means[label]), seg.segment_labels)
    @test seg.image_indexmap == expected

    expected = ones(Int, 5, 5)
    expected[2:4,2:4] .= 2
    expected[3,3] = 1
    expected_labels = [1,2]
    expected_means = Dict(1=>Gray{N0f8}(0.0), 2=>Gray{N0f8}(1.0))
    expected_count = Dict(1=>17, 2=>8)

    seg = seeded_region_growing(img, seeds, (5,5))
    @test all(label->(label in expected_labels), seg.segment_labels)
    @test all(label->(label in seg.segment_labels), expected_labels)
    @test expected_count == seg.segment_pixel_count
    @test expected_means == seg.segment_means
    @test seg.image_indexmap == expected

    # element-type and InexactError
    img = zeros(Int, 4, 4)
    img[2:end, 2:end] .= 10
    img[end,end] = 11
    seeds = [(CartesianIndex(1,1), 1), (CartesianIndex(2,2), 2)]

    expected = ones(Int, size(img))
    expected[2:end,2:end] .= 2
    expected_labels = [1,2]
    expected_means = Dict(1=>0.0, 2=>91/9)
    expected_count = Dict(1=>7, 2=>9)

    seg = seeded_region_growing(img, seeds, (3,3), (v1,v2)->(δ = abs(v1-v2); return δ > 1 ? δ : zero(δ)))
    @test all(label->(label in expected_labels), seg.segment_labels)
    @test all(label->(label in seg.segment_labels), expected_labels)
    @test expected_count == seg.segment_pixel_count
    @test expected_means == seg.segment_means
    @test seg.image_indexmap == expected

    # 3-d image
    img = zeros(RGB{N0f8},(9,9,9))
    img[3:7,3:7,3:7] .= RGB{N0f8}(0.5,0.5,0.5)
    img[2:5,5:9,4:6] .= RGB{N0f8}(0.8,0.8,0.8)
    seeds = [(CartesianIndex(1,1,1),1), (CartesianIndex(6,4,4),2), (CartesianIndex(3,6,5),3)]

    expected = ones(Int, (9,9,9))
    expected[3:7,3:7,3:7] .= 2
    expected[2:5,5:9,4:6] .= 3
    expected_labels = [1,2,3]
    expected_means = Dict([(i, of_mean_type(img[seeds[i][1]])) for i in 1:3])
    expected_count = Dict(1=>571, 2=>98, 3=>60)

    seg = seeded_region_growing(img, seeds)
    @test all(label->(label in expected_labels), seg.segment_labels)
    @test all(label->(label in seg.segment_labels), expected_labels)
    @test expected_count == seg.segment_pixel_count
    @test expected_means == seg.segment_means
    @test seg.image_indexmap == expected

    # custom diff_fn
    img = zeros(RGB{N0f8},(3,3))
    img[1:3,1] .= RGB{N0f8}(0.4,1,0)
    img[1:3,2] .= RGB{N0f8}(0.2,1,0)
    seeds = [(CartesianIndex(2,1),1), (CartesianIndex(2,3),2)]

    expected = ones(Int, (3,3))
    expected[1:3,3] .= 2
    expected_labels = [1,2]
    expected_means = Dict(1=>RGB{Float64}(0.3,1.0,0.0), 2=>RGB{Float64}(0.0,0.0,0.0))
    expected_count = Dict(1=>6, 2=>3)

    seg = seeded_region_growing(img, seeds)
    @test all(label->(label in expected_labels), seg.segment_labels)
    @test all(label->(label in seg.segment_labels), expected_labels)
    @test expected_count == seg.segment_pixel_count
    @test expected_means == seg.segment_means
    @test seg.image_indexmap == expected

    expected = ones(Int, (3,3))
    expected[1:3,2] .= 0
    expected[1:3,3] .= 2
    expected_labels = [0,1,2]
    expected_means = Dict(1=>RGB{Float64}(0.4,1.0,0.0), 2=>RGB{Float64}(0.0,0.0,0.0))
    expected_count = Dict(0=>3, 1=>3, 2=>3)

    seg = seeded_region_growing(img, seeds, [3,3], (c1,c2)->abs(of_mean_type(c1).r - of_mean_type(c2).r))
    @test all(label->(label in expected_labels), seg.segment_labels)
    @test all(label->(label in seg.segment_labels), expected_labels)
    @test expected_count == seg.segment_pixel_count
    @test expected_means == seg.segment_means
    @test seg.image_indexmap == expected
end

@testset "Unseeded Region Growing" begin
    # 2-D image
    img = zeros(Gray{N0f8}, 10, 10)
    img[6:10,4:8] .= 0.5
    img[3:7,2:6] .= 0.8

    expected = ones(Int, 10, 10)
    expected[6:10,4:8] .= 2
    expected[3:7,2:6] .= 3
    expected_labels = [1,2,3]
    expected_means = Dict(1 => of_mean_type(img[3,9]), 3 => of_mean_type(img[5,2]), 2 => of_mean_type(img[9,7]))
    expected_count = Dict(1 => 56, 3 => 25, 2 => 19)

    seg = unseeded_region_growing(img, 0.2)
    @test all(label->(label in expected_labels), seg.segment_labels)
    @test all(label->(label in seg.segment_labels), expected_labels)
    @test expected_count == seg.segment_pixel_count
    @test expected_means == seg.segment_means
    @test seg.image_indexmap == expected

    # Custom neighbourhood using a function
    seg = unseeded_region_growing(img, 0.2, c->[CartesianIndex(c[1]-1,c[2]), CartesianIndex(c[1]+1,c[2]), CartesianIndex(c[1],c[2]-1), CartesianIndex(c[1],c[2]+1)])
    @test all(label->(label in expected_labels), seg.segment_labels)
    @test all(label->(label in seg.segment_labels), expected_labels)
    @test expected_count == seg.segment_pixel_count
    @test expected_means == seg.segment_means
    @test seg.image_indexmap == expected

    # Offset image
    img = centered(img)
    expected = centered(expected)
    seg = unseeded_region_growing(img, 0.2)
    @test all(label->(label in expected_labels), seg.segment_labels)
    @test all(label->(label in seg.segment_labels), expected_labels)
    @test expected_count == seg.segment_pixel_count
    @test expected_means == seg.segment_means
    @test seg.image_indexmap == expected

    # Custom neighbourhood using a [5,5] kernel and varying threshold
    img = zeros(Gray{N0f8}, 5, 5)
    img[2:4,2:4] .= 1
    img[3,3] = 0.8

    expected = fill(1,(5,5))
    expected[2:4,2:4] .= 2
    expected_labels = [1,2]
    expected_means = Dict(1=>Gray{Float64}(0.0), 2=>Gray{Float64}(8.8/9))
    expected_count = Dict(2=>9, 1=>16)

    seg = unseeded_region_growing(img, 0.2, (5,5))
    @test all(label->(label in expected_labels), seg.segment_labels)
    @test all(label->(label in seg.segment_labels), expected_labels)
    @test expected_count == seg.segment_pixel_count
    @test all(label->(expected_means[label] ≈ seg.segment_means[label]), seg.segment_labels)
    @test seg.image_indexmap == expected

    expected = ones(Int, 5, 5)
    expected[2:4,2:4] .= 3
    expected[3,3] = 2
    expected_labels = [1,2,3]
    expected_means = Dict(1=>Gray{N0f8}(0.0), 3=>Gray{N0f8}(1.0), 2=>Gray{N0f8}(0.8))
    expected_count = Dict(1=>16, 2=>1, 3=>8)

    seg = unseeded_region_growing(img, 0.1, (5,5))
    @test all(label->(label in expected_labels), seg.segment_labels)
    @test all(label->(label in seg.segment_labels), expected_labels)
    @test expected_count == seg.segment_pixel_count
    @test expected_means == seg.segment_means
    @test seg.image_indexmap == expected

    expected = ones(Int, 5, 5)
    expected[2:4,2:4] .= 2
    expected[3,3] = 3
    expected_labels = [1,2,3]
    expected_means = Dict(1=>Gray{N0f8}(0.0), 2=>Gray{N0f8}(1.0), 3=>Gray{N0f8}(0.8))
    expected_count = Dict(1=>16, 3=>1, 2=>8)

    seg = unseeded_region_growing(img, 0.1, (3,3))
    @test all(label->(label in expected_labels), seg.segment_labels)
    @test all(label->(label in seg.segment_labels), expected_labels)
    @test expected_count == seg.segment_pixel_count
    @test expected_means == seg.segment_means
    @test seg.image_indexmap == expected

    # 3-d image
    img = zeros(RGB{N0f8},(9,9,9))
    img[3:7,3:7,3:7] .= RGB{N0f8}(0.5,0.5,0.5)
    img[2:5,5:9,4:6] .= RGB{N0f8}(0.8,0.8,0.8)

    expected = ones(Int, (9,9,9))
    expected[3:7,3:7,3:7] .= 2
    expected[2:5,5:9,4:6] .= 3
    expected_labels = [1,2,3]
    expected_means = Dict(1=>of_mean_type(img[1,1,1]), 2=>of_mean_type(img[3,3,3]), 3=>of_mean_type(img[2,5,4]))
    expected_count = Dict(1=>571, 2=>98, 3=>60)

    seg = unseeded_region_growing(img, 0.2)
    @test all(label->(label in expected_labels), seg.segment_labels)
    @test all(label->(label in seg.segment_labels), expected_labels)
    @test expected_count == seg.segment_pixel_count
    @test expected_means == seg.segment_means
    @test seg.image_indexmap == expected

    # custom diff_fn
    img = zeros(RGB{N0f8},(3,3))
    img[1:3,1] .= RGB{N0f8}(0.4,1,0)
    img[1:3,2] .= RGB{N0f8}(0.2,1,0)

    expected = ones(Int, (3,3))
    expected[1:3,2] .= 2
    expected[1:3,3] .= 3
    expected_labels = [1,2,3]
    expected_means = Dict(1=>of_mean_type(img[1,1]), 3=>of_mean_type(img[1,3]), 2=>of_mean_type(img[1,2]))
    expected_count = Dict(1=>3, 2=>3, 3=>3)

    seg = unseeded_region_growing(img, 0.2, [3,3], (c1,c2)->abs(of_mean_type(c1).r - of_mean_type(c2).r))
    @test all(label->(label in expected_labels), seg.segment_labels)
    @test all(label->(label in seg.segment_labels), expected_labels)
    @test expected_count == seg.segment_pixel_count
    @test expected_means == seg.segment_means
    @test seg.image_indexmap == expected

end
