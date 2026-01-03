# volk-rs

Rust bindings for the volk library.

https://github.com/ThomasHabets/volk-rs

## What is VOLK

[VOLK][volk] is the Vector-Optimized Library of Kernels. It is a library that
contains kernels of hand-written SIMD code for different mathematical
operations. Since each SIMD architecture can be very different and no compiler
has yet come along to handle vectorization properly or highly efficiently, VOLK
approaches the problem differently.

For each architecture or platform that a developer wishes to vectorize for, a
new proto-kernel is added to VOLK. At runtime, VOLK will select the correct
proto-kernel. In this way, the users of VOLK call a kernel for performing the
operation that is platform/architecture agnostic. This allows us to write
portable SIMD code.

## What is this crate

This crate adds safe Rust bindings for volk. It also provides unsafe wrappers,
to minimize overhead for the adventurous.

## Supported kernels in this Rust wrapper

Kernels are currently added on demand. A TODO is to just go through and add them
all.

[volk]: https://github.com/gnuradio/volk
