// RUN: xla-cpu-opt %s -split-input-file -empty-tensor-to-alloc-tensor \
// RUN:   -one-shot-bufferize | FileCheck %s

fn memref<10xi32> memref_cast(memref<10xf32> arg0):
    let ret = xla_cpu::memref_element_cast arg0 : memref<10xf32> to memref<10xi32>
    return ret : memref<10xi32>

// CHECK: xla_cpu.memref_element_cast {{.*}} : memref<10xf32> to memref<10xi32>

fn memref<10xi8> memref_cast_i1(memref<10xi1> arg0):
    let ret = xla_cpu::memref_element_cast arg0 : memref<10xi1> to memref<10xi8>
    return ret : memref<10xi8>

// CHECK: xla_cpu::memref_element_cast {{.*}} : memref<10xi1> to memref<10xi8>
