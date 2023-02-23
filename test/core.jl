@testset "core" begin

  # Accessor functions
  img = fill(1.0, (4,4))
  img[1:2,1:2] .= 2.0
  img[1:2,3:4] .= 3.0
  seg = fast_scanning(img, 0.5)

  @test labels_map(seg) == seg.image_indexmap
  @test segment_labels(seg) == seg.segment_labels
  @test segment_mean(seg) == seg.segment_means
  @test all([segment_mean(seg, i) == seg.segment_means[i] for i in keys(seg.segment_means)])
  @test segment_pixel_count(seg) == seg.segment_pixel_count
  @test all([segment_pixel_count(seg, i) == seg.segment_pixel_count[i] for i in keys(seg.segment_pixel_count)])

  img = [1 1.1 0.9 10 10.2 9.8; 1 1 1.1 4 3.8 4]
  r = fuzzy_cmeans(img, 2, 2.0)

  @test segment_labels(r) == [1,2]
  @test segment_mean(r, 1) == r.centers[:,1]
  @test segment_mean(r) == Dict(1=>r.centers[:,1], 2=>r.centers[:,2])
  @test all([segment_pixel_count(r,i) == 6 for i in 1:2])
  @test segment_pixel_count(r) == Dict(1=>6, 2=>6)

  # RAG
  img = fill(1.0, (10,10))
  img[:, 5:10] .= 2.0
  seg = fast_scanning(img, 0.5)
  g, vm = region_adjacency_graph(seg, (i,j)->sum(abs2, seg.segment_means[i]-seg.segment_means[j]))

  expectedg = SimpleWeightedGraph(2)
  add_edge!(expectedg, 1, 2, 1.0)
  expectedvm = Dict(1=>1, 2=>2)

  @test g == expectedg
  @test vm == expectedvm

  img = fill(1.0, (10,10))
  img[3:8,3:8] .= 2.0
  img[5:7,5:7] .= 3.0
  seg = fast_scanning(img, 0.5)
  g, vm = region_adjacency_graph(seg, (i,j)->1)

  expectedg = SimpleWeightedGraph(3)
  add_edge!(expectedg,1,2)
  add_edge!(expectedg,2,3)
  expectedvm = Dict(1=>1, 2=>2, 3=>3)

  @test g == expectedg
  @test vm == expectedvm

  # remove_segment
  img = fill(1.0, (10,10))
  img[1:4,:] .= 2.0
  img[:,5:10] .= 4.0
  seg = fast_scanning(img, 0.5)
  new_seg = remove_segment(seg, 1, (i,j)->(-seg.segment_pixel_count[j]))

  expected = fill(3, (10,10))
  expected[5:10,1:4] .= 2
  expected_labels = [2,3]
  expected_means = Dict(2=>1.0, 3=>68/19)
  expected_count = Dict(2=>24, 3=>76)

  @test all(label->(label in expected_labels), new_seg.segment_labels)
  @test all(label->(label in new_seg.segment_labels), expected_labels)
  @test expected_count == new_seg.segment_pixel_count
  @test all(label->(expected_means[label] ≈ new_seg.segment_means[label]), new_seg.segment_labels)
  @test new_seg.image_indexmap == expected

  new_seg = remove_segment(seg, 1, (i,j)->sum(abs2, seg.segment_means[i]-seg.segment_means[j]))

  expected = fill(3, (10,10))
  expected[:,1:4] .= 2
  expected_means = Dict(2=>1.4, 3=>4.0)
  expected_count = Dict(2=>40, 3=>60)

  @test all(label->(label in expected_labels), new_seg.segment_labels)
  @test all(label->(label in new_seg.segment_labels), expected_labels)
  @test expected_count == new_seg.segment_pixel_count
  @test all(label->(expected_means[label] ≈ new_seg.segment_means[label]), new_seg.segment_labels)
  @test new_seg.image_indexmap == expected

  remove_segment!(seg, 1, (i,j)->sum(abs2, seg.segment_means[i]-seg.segment_means[j]))

  @test all(label->(label in expected_labels), seg.segment_labels)
  @test all(label->(label in seg.segment_labels), expected_labels)
  @test expected_count == seg.segment_pixel_count
  @test all(label->(expected_means[label] ≈ seg.segment_means[label]), seg.segment_labels)
  @test seg.image_indexmap == expected

  # prune_segments
  img = fill(1.0, (10,10))
  img[3,3] = 2.0
  img[5,5] = 2.0
  img[8:9,7:8] .= 3.0
  seg = fast_scanning(img, 0.5)
  new_seg = prune_segments(seg, l->(seg.segment_pixel_count[l] < 2), (i,j)->sum(abs2, seg.segment_means[i]-seg.segment_means[j]))

  expected = fill(1, (10,10))
  expected[8:9,7:8] .= 4
  expected_labels = [1,4]
  expected_means = Dict(4=>3.0, 1=>49/48)
  expected_count = Dict(4=>4, 1=>96)

  @test all(label->(label in expected_labels), new_seg.segment_labels)
  @test all(label->(label in new_seg.segment_labels), expected_labels)
  @test expected_count == new_seg.segment_pixel_count
  @test all(label->(expected_means[label] ≈ new_seg.segment_means[label]), new_seg.segment_labels)
  @test new_seg.image_indexmap == expected

  new_seg = prune_segments(seg, [2,3], (i,j)->sum(abs2, seg.segment_means[i]-seg.segment_means[j]))

  @test all(label->(label in expected_labels), new_seg.segment_labels)
  @test all(label->(label in new_seg.segment_labels), expected_labels)
  @test expected_count == new_seg.segment_pixel_count
  @test all(label->(expected_means[label] ≈ new_seg.segment_means[label]), new_seg.segment_labels)
  @test new_seg.image_indexmap == expected

  # region_adjacency_graph
  img = fill(1.0, (10,10))
  img[4:6, 4:6] .= 0

  weight_fn(I, J) = exp(-abs(I.second - J.second)) / ((I.first[1]-J.first[1])^2 + (I.first[2]-J.first[2])^2)
  G, vertex2cartesian = region_adjacency_graph(img, weight_fn, 2)

  @test nv(G) == 100
  @test G.weights[45,46] == 1
  @test G.weights[43,44] == exp(-1)

  #test that all edges weights equal weights computed using weight_fn
  @test all(map(edge-> edge.weight == weight_fn(vertex2cartesian[edge.src]=>img[vertex2cartesian[edge.src]], vertex2cartesian[edge.dst]=>img[vertex2cartesian[edge.dst]]), edges(G)))

  #test that all edges are between nodes less than r pixels away
  @test all(map(edge-> max(abs(vertex2cartesian[edge.src][1] - vertex2cartesian[edge.dst][1]), abs(vertex2cartesian[edge.src][2] - vertex2cartesian[edge.dst][2])) <= 2 , edges(G)))

  @test region_adjacency_graph(img, weight_fn, 2) == region_adjacency_graph(img, weight_fn, CartesianIndex(2,2))

  G, vertex2cartesian = region_adjacency_graph(img, weight_fn, CartesianIndex(1,2))
  @test G.weights[45,47] == 0
  @test G.weights[45,65] != 0

  #test box_iterator
  fiter = ImageSegmentation.box_iterator((3, 3))
  center = CartesianIndex(17, 24)
  @test isa(fiter(center), CartesianIndices)
  @test minimum(fiter(center)) == CartesianIndex(16, 23)
  @test maximum(fiter(center)) == CartesianIndex(18, 25)
  
end
