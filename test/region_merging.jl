homogeneous(i::AbstractArray{T,N}) where {T<:Union{Colorant,Real}, N} = -(extrema(i)...) == 0

@testset "Region Splitting" begin

  # 2-d image
  img = fill(1, (8,8))
  img[1:4,1:4] = 2

  t = region_tree(img, homogeneous)

  @test t[1,1].data == ((2,16))
  @test t[1,2].data == ((1,16))
  @test t[2,1].data == ((1,16))
  @test t[2,2].data == ((1,16))
  @test all(isleaf, [t[1,1], t[1,2], t[2,1], t[2,2]])

  seg = region_splitting(img, homogeneous)

  expected = fill(1,(8,8))
  expected[5:8,1:4] = 2
  expected[1:4,5:8] = 3
  expected[5:8,5:8] = 4
  expected_labels = [1,2,3,4]
  expected_means = Dict(1=>2, 2=>1, 3=>1, 4=>1)
  expected_count = Dict(1=>16, 2=>16, 3=>16, 4=>16)

  @test all(label->(label in expected_labels), seg.segment_labels)
  @test all(label->(label in seg.segment_labels), expected_labels)
  @test expected_count == seg.segment_pixel_count
  @test all(label->(expected_means[label] ≈ seg.segment_means[label]), seg.segment_labels)
  @test seg.image_indexmap == expected


  # 3-d image
  img = fill(1, (4,4,4))
  img[3:4,3:4,3:4] = 2

  seg = region_splitting(img, homogeneous)

  expected = fill(1,(4,4,4))
  expected[3:4,1:2,1:2] = 2
  expected[1:2,3:4,1:2] = 3
  expected[3:4,3:4,1:2] = 4
  expected[1:2,1:2,3:4] = 5
  expected[3:4,1:2,3:4] = 6
  expected[1:2,3:4,3:4] = 7
  expected[3:4,3:4,3:4] = 8
  expected_labels = [1,2,3,4,5,6,7,8]
  expected_means = Dict(ntuple(i->(i=>1),7)..., 8=>2)
  expected_count = Dict(ntuple(i->(i=>8), 8)...)

  @test all(label->(label in expected_labels), seg.segment_labels)
  @test all(label->(label in seg.segment_labels), expected_labels)
  @test expected_count == seg.segment_pixel_count
  @test all(label->(expected_means[label] ≈ seg.segment_means[label]), seg.segment_labels)
  @test seg.image_indexmap == expected

end
