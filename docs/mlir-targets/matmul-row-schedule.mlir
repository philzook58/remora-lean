module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %fill = transform.structured.match ops{["linalg.fill"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %epilogue = transform.structured.match ops{["linalg.generic"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %tiled, %outer = transform.structured.tile_using_for %epilogue tile_sizes [1, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused, %loop = transform.structured.fuse_into_containing_op %matmul into %outer
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %filled, %final_loop = transform.structured.fuse_into_containing_op %fill into %loop
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
