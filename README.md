# OpenBLAS Bindings

The API is designed around ILP64. There is the `Blas_Int` type I eventually plan to place on a `#configure` to enable `LP32` functionality. Note that the ILP64 is a different package on the package managers than the basic version.

Provides a complete set of bindings to [OpenBLAS 0.3.30](https://github.com/OpenMathLib/OpenBLAS/releases/tag/v0.3.30). 

## Status

This library is in the *early* stages of design and should be considered unstable (and likely has errors). Raw bindings were produced with Odin-Bindgen, and have mostly not been cleaned up (eg ^ to [^] and other similar edits).

Files at the top level build okay, but need the design changes noted below. Files in the `_UNTESTED` folder are not yet cleaned up.

## Dependancy

Windows: Download the x64 library, link against the .lib file and include the dll alongside your main application
Linux: `sudo apt install libopenblas64-dev`
Darwin: may need to build openblas from source; could try `brew install openblas --with-ilp64` (unverified)

## Design

Raw bindings are located in `c/` and `f77/`.

All bindings are wrapped in `Matrix($T)` and `Vector($T)` providing a higher level interface than the raw bindings. You may use these, or just the raw bindings.

The library currently allocates internally in the wrapped calls, though its my goal to redesign the api roughly as follows:

```odin
// Current Design (100% internal allocs)
m_svd_f32_c64 :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	compute_u: bool = true,
	compute_vt: bool = true,
	full_matrices: bool = false,
	allocator := context.allocator,
) -> (
	S: []f32,
	U: Matrix(T),
	VT: Matrix(T),
	info: Info, 
) where T == f32 || T == complex64

// Planned Design (No to minimal internal allocs):

// Find the Optimal Workspace:
bufReq := query_workspace_svd(&A, compute_u=true, compute_v=true) 
buf:= make([]u8, bufReq) // user allocates
U:Matrix(f64)
VT:Matrix(f64)
S:Vector(f64)
// Make Compute Call:
m_svd(&A,&U,&VT,&S,compute_u=true, compute_v=true,&buf)
```