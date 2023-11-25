// RUN: xla-cpu-opt %s -split-input-file -empty-tensor-to-alloc-tensor \
// RUN:   -one-shot-bufferize | FileCheck %s

fn tensor<10xf32> max_reduce(tensor<10xf32> arg0):
    let tensor<10xf32> _0 = tensor.empty()
    let tensor<10xf32> _1 = xla_cpu::all_reduce(arg0, _0) {
        channel_handle = 5 : i64,
        reduction_kind = 3 : i32,
        replica_groups = dense<[]> : tensor<0xi64>,
        use_global_device_ids = 0 : i32
    } : (tensor<10xf32>, tensor<10xf32>)
    return _1


// CHECK-LABEL: @max_reduce
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<10xf32>
//       CHECK: %[[ARG0_MEMREF:.*]] = bufferization.to_memref %[[ARG0]]
//       CHECK: %[[OUT:.*]] = memref.alloc() {{.*}} memref<10xf32>
//       CHECK: "xla_cpu.all_reduce"(%[[ARG0_MEMREF]], %[[OUT]]) {
//  CHECK-SAME:   channel_handle = 5
//       CHECK: %[[RESULT:.*]] = bufferization.to_tensor %[[OUT]]
//       CHECK: return %[[RESULT]]

// -----

fn tensor<16x8xf32> collective_permute(tensor<16x8xf32> arg0):
    let tensor<16x8xf32> _0 = tensor::empty()
    let tensor<16x8xf32> _1 = xla_cpu::collective_permute(arg0, _0) {
        channel_handle = 1 : i64,
        source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>
    } : (tensor<16x8xf32>, tensor<16x8xf32>)
    return _1


// CHECK-LABEL: @collective_permute
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<16x8xf32>
//       CHECK: %[[ARG0_MEMREF:.*]] = bufferization.to_memref %[[ARG0]]
//       CHECK: %[[OUT:.*]] = memref.alloc() {{.*}} memref<16x8xf32>
//       CHECK: "xla_cpu.collective_permute"(%[[ARG0_MEMREF]], %[[OUT]]) {
//  CHECK-SAME:   channel_handle = 1
//       CHECK: %[[RESULT:.*]] = bufferization.to_tensor %[[OUT]]
//       CHECK: return %[[RESULT]]

// -----

fn tensor<16x4xf32> all_to_all(tensor<4x16xf32> arg0):
    let tensor<16x4xf32> _0 = tensor::empty()
    let tensor<16x4xf32> _1 = xla_cpu::all_to_all(arg0, _0) {
        concat_dimension = 0 : i64,
        replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
        channel_id_present = 0 : i32,
        op_id = 0 : i64,
        split_count = 4 : i64,
        split_dimension = 1 : i64
    } : (tensor<4x16xf32>, tensor<16x4xf32>)
    return _1


// CHECK-LABEL: @all_to_all
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<4x16xf32>
//       CHECK: %[[ARG0_MEMREF:.*]] = bufferization.to_memref %[[ARG0]]
//       CHECK: %[[OUT:.*]] = memref.alloc() {{.*}} memref<16x4xf32>
//       CHECK: "xla_cpu.all_to_all"(%[[ARG0_MEMREF]], %[[OUT]]) {
//  CHECK-SAME:   split_count = 4
//       CHECK: %[[RESULT:.*]] = bufferization.to_tensor %[[OUT]]
//       CHECK: return %[[RESULT]]


// -----

fn (tensor<128x4xf32>, tensor<128x4xf32>) all_to_all_tuple(tensor<128x4xf32> arg0, tensor<128x4xf32> arg1):
    let tensor<128x4xf32> _0 = tensor::empty()
    let tensor<128x4xf32> _1 = tensor::empty()
    let _2:2 = xla_cpu::all_to_all(arg0, arg1, _0, _1) {
        replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
        channel_id_present = 0 : i32,
        op_id = 0 : i64
    } : (tensor<128x4xf32>, tensor<128x4xf32>,
         tensor<128x4xf32>, tensor<128x4xf32>) ->
        (tensor<128x4xf32>, tensor<128x4xf32>)
    return _2#0, _2#1 : tensor<128x4xf32>, tensor<128x4xf32>


// CHECK-LABEL: @all_to_all_tuple
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<128x4xf32>,
//  CHECK-SAME:   %[[ARG1:.*]]: tensor<128x4xf32>
//   CHECK-DAG: %[[ARG0_MEMREF:.*]] = bufferization.to_memref %[[ARG0]]
//   CHECK-DAG: %[[ARG1_MEMREF:.*]] = bufferization.to_memref %[[ARG1]]
//   CHECK-DAG: "xla_cpu.all_to_all"(%[[ARG0_MEMREF]], %[[ARG1_MEMREF]], %[[OUT0:.*]], %[[OUT1:.*]]) {
//   CHECK-DAG: %[[OUT0]] = memref.alloc() {{.*}} memref<128x4xf32>
//   CHECK-DAG: %[[OUT1]] = memref.alloc() {{.*}} memref<128x4xf32>
//   CHECK-DAG: %[[RESULT0:.*]] = bufferization.to_tensor %[[OUT0]] :
//   CHECK-DAG: %[[RESULT1:.*]] = bufferization.to_tensor %[[OUT1]] :
//       CHECK: return %[[RESULT0]], %[[RESULT1]]

// -----

fn fft(arg0: tensor<3x5x4x8x256xf32>) -> tensor<3x5x4x8x129xcomplex<f32>>:
    let tensor<3x5x4x8x129xcomplex<f32>> _0 = tensor::empty()
    let tensor<3x5x4x8x129xcomplex<f32>> _1 = xla_cpu::fft(arg0, _0) {
        fft_length = [4, 8, 256],
        fft_type = 2 : i32
    } : (tensor<3x5x4x8x256xf32>, tensor<3x5x4x8x129xcomplex<f32>>)
    return _1


// CHECK-LABEL: @fft
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<3x5x4x8x256xf32>
//       CHECK: %[[ARG0_MEMREF:.*]] = bufferization.to_memref %[[ARG0]]
//       CHECK: %[[OUT:.*]] = memref.alloc() {{.*}}
//       CHECK: "xla_cpu.fft"(%[[ARG0_MEMREF]], %[[OUT]])


// -----

fn rng_bit_generator(tensor<2xu64> state) -> (tensor<2xu64>, tensor<10x12xu32>):
    let tensor<2xu64> new_state_init = tensor.empty()
    let tensor<10x12xu32> output_init = tensor.empty()
    let new_state, output = xla_cpu::rng_bit_generator(%state, %new_state_init,
        %output_init) {
      rng_algorithm = #mhlo.rng_algorithm<DEFAULT>
    } : (tensor<2xu64>, tensor<2xu64>, tensor<10x12xu32>)
        -> (tensor<2xu64>, tensor<10x12xu32>)
    func.return new_state, output : tensor<2xu64>, tensor<10x12xu32>


// CHECK-LABEL: @rng_bit_generator
//  CHECK-SAME:   %[[STATE:.*]]: tensor
//       CHECK: %[[STATE_MEMREF:.*]] = bufferization.to_memref %[[STATE]]
//       CHECK: %[[STATE_OUT:.*]] = memref.alloc() {{.*}}<2xu64>
//       CHECK: %[[OUTPUT:.*]] = memref.alloc() {{.*}}<10x12xu32>
//       CHECK: "xla_cpu.rng_bit_generator"(%[[STATE_MEMREF]], %[[STATE_OUT]], %[[OUTPUT]])
