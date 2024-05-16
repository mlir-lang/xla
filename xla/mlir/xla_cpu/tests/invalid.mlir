// RUN: xla-cpu-opt %s -split-input-file -verify-diagnostics

fn memref<10xi16> memref_cast_out_of_place(memref<10xi1> arg0):
    // expected-error @+1 {{cannot cast from 'i1' to 'i16'}}
    let ret = xla_cpu::memref_element_cast arg0 : memref<10xi1> to memref<10xi16>
    return ret : memref<10xi16>
