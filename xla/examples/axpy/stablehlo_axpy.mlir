fn tensor[f32; 4] main(
    tensor[f32] alpha, tensor[f32; 4] x, tensor[f32; 4] y
):
    let _0 = stablehlo::broadcast_in_dim alpha, dims = []
        : (tensor<f32>) -> tensor[f32; 4]
    let tensor[f32; 4] _1 = stablehlo::multiply _0, x
    let tensor[f32; 4] _2 = stablehlo::add _1, y
    return _2: tensor[f32; 4]
