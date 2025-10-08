# OpenBLAS Bindings

The API is designed around ILP64. There is the `Blas_Int` type I eventually plan to place on a `#configure` to enable `LP32` functionality. Note that the ILP64 is a different package on the package managers than the basic version.

Provides a complete set of bindings to [OpenBLAS 0.3.30](https://github.com/OpenMathLib/OpenBLAS/releases/tag/v0.3.30). 

## Status

This library is in the *early* stages of design and should be considered unstable (and likely has errors). Raw bindings were produced with Odin-Bindgen, and subsequently cleaned up.

I believe I have most/all of the public LAPACK and BLAS APIs wrapped, however some types are not yet using the type specific wrapper structs.

Due to conditional compilation, it is likely that there are compiler errors I haven't yet found. Adding tests / verifying the wrappers is an ongoing activity.

## Dependancy

Windows: Download the x64 library, link against the .lib file and include the dll alongside your main application

Linux: `sudo apt install libopenblas64-dev`

Darwin: may need to build openblas from source; could try `brew install openblas --with-ilp64` (unverified)

## Design

Raw bindings are located in `c/` and `f77/`.

All bindings are wrapped in `Matrix($T)` and `Vector($T)` providing a higher level interface than the raw bindings. You may use these, or just the raw bindings.

Reference `LAPACK_FUNCTION_TABLE.md` for a map of function and wrapping types

## Usage

The API requires pre-allocated arrays and separates workspace queries from computation:

```odin
package example

import ob "../openblas"

// Example: Solve a banded linear system using LU factorization
solve_banded_system :: proc() {
    // Problem dimensions
    n := 100      // Matrix size
    kl := 2       // Lower bandwidth
    ku := 3       // Upper bandwidth
    nrhs := 1     // Number of right-hand sides

    // Step 1: Query workspace and result sizes
    ipiv_size := ob.query_result_sizes_solve_banded(n)

    // Step 2: Allocate arrays
    ipiv := make([]ob.Blas_Int, ipiv_size)
    defer delete(ipiv)

    // Step 3: Create banded matrix and right-hand side
    AB := ob.create_banded_matrix(f64, n, n, kl, ku)
    defer ob.destroy_matrix(&AB)

    B := ob.create_matrix(f64, n, nrhs)
    defer ob.destroy_matrix(&B)

    // ... initialize AB and B with the data ...

    // Step 4: Solve the system (AB is overwritten with LU factorization)
    info, ok := ob.solve_banded(&AB, &B, ipiv)
    if !ok {
        fmt.printf("Solve failed with info = %d\n", info)
        return
    }

    // Solution is now in B
}
```

### API Usage Patterns

1. **Query Functions**: Call `query_workspace_*` and `query_result_sizes_*` first to determine workspace and result sizes
2. **Pre-allocation**: Allocate all arrays before calling solve procedures
3. **No Auto-allocation**: Procedures do not allocate memory internally


## Accelerate Compatibility

Mac is usually setup for LP32.. it can use ILP64, but the bindings have a `_64` suffix. We'll need to make a conditional foreign file that does this something along the lines of `USE_ACCERERATE && USE_ILP64{foriegn {...}}`