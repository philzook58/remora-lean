// Handwritten target: 13x13 max-pool with explicit affine window accesses.
// This is equivalent to the demo helper, not yet emitted automatically.
module {
func.func @main(%input: tensor<512x19x19xf32>) -> tensor<512x19x19xf32> {
 %negative = arith.constant -1.0e30 : f32
 %padded = tensor.pad %input low[0, 6, 6] high[0, 6, 6] {
 ^bb0(%c: index, %y: index, %x: index):
  tensor.yield %negative : f32
 } : tensor<512x19x19xf32> to tensor<512x31x31xf32>
 %window = arith.constant dense<0.0> : tensor<13x13xf32>
 %empty = tensor.empty() : tensor<512x19x19xf32>
 %init = linalg.fill ins(%negative : f32) outs(%empty : tensor<512x19x19xf32>) -> tensor<512x19x19xf32>
 %result = linalg.generic {
  indexing_maps = [affine_map<(c,y,x,ky,kx)->(c,y+ky,x+kx)>, affine_map<(c,y,x,ky,kx)->(ky,kx)>, affine_map<(c,y,x,ky,kx)->(c,y,x)>],
  iterator_types = ["parallel","parallel","parallel","reduction","reduction"]
 } ins(%padded, %window : tensor<512x31x31xf32>, tensor<13x13xf32>) outs(%init : tensor<512x19x19xf32>) {
 ^bb0(%v: f32, %unused: f32, %acc: f32):
  %max = arith.maximumf %v, %acc : f32
  linalg.yield %max : f32
 } -> tensor<512x19x19xf32>
 return %result : tensor<512x19x19xf32>
}
}
