// Same identity gather, with an explicit index operand: standard fusion removes it.
module {
func.func @main(%data: tensor<32xf32>) -> tensor<32xf32> {
 %e = tensor.empty() : tensor<32xi64>
 %idx = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>], iterator_types = ["parallel"]} outs(%e : tensor<32xi64>) {
 ^bb0(%unused: i64):
  %i = linalg.index 0 : index
  %v = arith.index_cast %i : index to i64
  linalg.yield %v : i64
 } -> tensor<32xi64>
 %o = tensor.empty() : tensor<32xf32>
 %r = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]} ins(%idx : tensor<32xi64>) outs(%o : tensor<32xf32>) {
 ^bb0(%j: i64, %unused: f32):
  %k = arith.index_cast %j : i64 to index
  %v = tensor.extract %data[%k] : tensor<32xf32>
  linalg.yield %v : f32
 } -> tensor<32xf32>
 return %r : tensor<32xf32>
}
}
