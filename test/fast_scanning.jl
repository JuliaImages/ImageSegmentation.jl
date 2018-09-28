@testset "Fast scanning" begin
  # 2-D array
  img = zeros(Float64, 9, 9)
  img[4:6,:] .= 10
  img[:,4:6] .= 11

  seg = fast_scanning(img, 2)
  expected = fill(4,(9,9))
  expected[1:3,1:3] .= 1
  expected[1:3,7:9] .= 5
  expected[7:9,1:3] .= 3
  expected[7:9,7:9] .= 6
  expected_labels = [1,3,4,5,6]
  expected_means = Dict(1=>0.0,3=>0.0,4=>10.6,5=>0.0,6=>0.0)
  expected_count = Dict(1=>9,3=>9,4=>45,5=>9,6=>9)

  @test all(label->(label in expected_labels), seg.segment_labels)
  @test all(label->(label in seg.segment_labels), expected_labels)
  @test expected_count == seg.segment_pixel_count
  @test all(label->(expected_means[label] ≈ seg.segment_means[label]), seg.segment_labels)
  @test seg.image_indexmap == expected

  # Offset array
  img = centered(img)

  seg = fast_scanning(img, 2)
  expected = centered(expected)

  @test all(label->(label in expected_labels), seg.segment_labels)
  @test all(label->(label in seg.segment_labels), expected_labels)
  @test expected_count == seg.segment_pixel_count
  @test all(label->(expected_means[label] ≈ seg.segment_means[label]), seg.segment_labels)
  @test seg.image_indexmap == expected

  # 3-D array
  img = zeros(Float64, (7,7,7))
  img[3:5,2:6,1:3] .= 2
  img[1:4,5:7,2:6] .= 3
  img[3:5,3:5,3:5] .= 8

  seg = fast_scanning(img, 1.5)
  expected = fill(5, (7,7,7))
  expected[3:5,2:6,1:3] .= 3
  expected[1:4,5:7,2:6] .= 3
  expected[3:5,3:5,3:5] .= 4
  expected_labels = [3,4,5]
  expected_means = Dict(3=>222/84,4=>8.0,5=>0.0)
  expected_count = Dict(3=>84,4=>27,5=>232)

  @test all(label->(label in expected_labels), seg.segment_labels)
  @test all(label->(label in seg.segment_labels), expected_labels)
  @test expected_count == seg.segment_pixel_count
  @test all(label->(expected_means[label] ≈ seg.segment_means[label]), seg.segment_labels)
  @test seg.image_indexmap == expected

  # custom diff_fn
  img = zeros(RGB{N0f8}, (3,3))
  img[2,:] .= RGB{N0f8}(0.2, 0.4, 0.4)
  img[3,:] .= RGB{N0f8}(0.2, 1.0, 1.0)

  seg = fast_scanning(img, 0.22)
  expected = ones(Int, (3,3))
  expected[2,:] .= 2
  expected[3,:] .= 3
  expected_labels = [1,2,3]
  expected_means = Dict(1=>RGB{Float64}(0.0,0.0,0.0),2=>RGB{Float64}(0.2,0.4,0.4),3=>RGB{Float64}(0.2,1.0,1.0))
  expected_count = Dict(1=>3,2=>3,3=>3)

  @test all(label->(label in expected_labels), seg.segment_labels)
  @test all(label->(label in seg.segment_labels), expected_labels)
  @test expected_count == seg.segment_pixel_count
  @test all(label->(expected_means[label] ≈ seg.segment_means[label]), seg.segment_labels)
  @test seg.image_indexmap == expected

  seg = fast_scanning(img, 0.22, (i,j)->(abs(i.r-j.r)))
  expected = ones(Int, (3,3))
  expected_labels = [1]
  expected_means = Dict(1=>RGB{Float64}(1.2/9,4.2/9,4.2/9))
  expected_count = Dict(1=>9)

  @test all(label->(label in expected_labels), seg.segment_labels)
  @test all(label->(label in seg.segment_labels), expected_labels)
  @test expected_count == seg.segment_pixel_count
  @test all(label->(expected_means[label] ≈ seg.segment_means[label]), seg.segment_labels)
  @test seg.image_indexmap == expected

  # Adaptive Thresholding
  img = [ 0.0 0.1 0.7 0.9;
          0.0 0.1 0.7 0.9;
          0.2 0.3 0.7 0.9;
          0.2 0.3 0.7 0.9; ]

  seg = fast_scanning(img, [0.2 0.1; 0.1 0.3])
  expected =  [ 1 1 4 4;
                1 1 4 4;
                2 2 4 4;
                2 2 4 4; ]
  expected_labels = [1,2,4];
  expected_means = Dict(1=>0.05,2=>0.25,4=>0.8)
  expected_count = Dict(1=>4,2=>4,4=>8)

  @test all(label->(label in expected_labels), seg.segment_labels)
  @test all(label->(label in seg.segment_labels), expected_labels)
  @test expected_count == seg.segment_pixel_count
  @test all(label->(expected_means[label] ≈ seg.segment_means[label]), seg.segment_labels)
  @test seg.image_indexmap == expected

  seg = fast_scanning(img, (2,2))
  expected =  [ 1 3 5 6;
                1 3 5 6;
                2 4 5 6;
                2 4 5 6; ]
  expected_labels = [1,2,3,4,5,6]
  expected_means = Dict(1=>0.0,2=>0.2,3=>0.1,4=>0.3,5=>0.7,6=>0.9)
  expected_count = Dict(1=>2,2=>2,3=>2,4=>2,5=>4,6=>4)

  @test all(label->(label in expected_labels), seg.segment_labels)
  @test all(label->(label in seg.segment_labels), expected_labels)
  @test expected_count == seg.segment_pixel_count
  @test all(label->(expected_means[label] ≈ seg.segment_means[label]), seg.segment_labels)
  @test seg.image_indexmap == expected
end
